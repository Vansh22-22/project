import streamlit as st
import google.generativeai as genai
import json
import hashlib
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

# Page configuration
st.set_page_config(
    page_title="AI Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #667eea;
        font-weight: 600;
        margin-top: 1rem;
    }
    .course-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .quiz-card {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-msg {
        background: #48bb78;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0

# Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_user_data(username):
    if username not in st.session_state.user_data:
        st.session_state.user_data[username] = {
            'courses': [],
            'quiz_history': [],
            'learning_preferences': [],
            'total_score': 0,
            'courses_completed': 0,
            'quizzes_taken': 0,
            'learning_streak': 0
        }

def configure_gemini():
    if st.session_state.api_key:
        try:
            genai.configure(api_key=st.session_state.api_key)
            return True
        except Exception as e:
            st.error(f"API Configuration Error: {str(e)}")
            return False
    return False

def test_api_key_and_list_models():
    """Test API key and list available models"""
    if not st.session_state.api_key:
        return None, "No API key provided"
    
    try:
        genai.configure(api_key=st.session_state.api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Extract just the model name without 'models/' prefix
                model_name = m.name.replace('models/', '')
                models.append(model_name)
        return models, None
    except Exception as e:
        return None, str(e)

def get_working_model():
    """Get the first working model, prioritizing Flash models for rate limits"""
    if 'working_model' in st.session_state and st.session_state.working_model:
        return st.session_state.working_model
    
    # Try to find working models
    available_models, error = test_api_key_and_list_models()
    
    if available_models:
        # Prefer Flash models (they have higher rate limits)
        preferred = [
            'gemini-1.5-flash-8b',  # Fastest, highest limits
            'gemini-1.5-flash',      # Fast, good limits
            'gemini-2.0-flash-exp',  # Experimental flash
            'gemini-1.5-pro',        # Slower but capable
            'gemini-pro'             # Fallback
        ]
        
        for pref in preferred:
            if pref in available_models:
                st.session_state.working_model = pref
                return pref
        
        # If no preferred model, use the first available
        if available_models:
            st.session_state.working_model = available_models[0]
            return available_models[0]
    
    return None

def rate_limit_check():
    """Simple rate limiting to avoid quota issues"""
    current_time = time.time()
    
    # Reset counter every minute
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time
    
    # Check if we've made too many requests
    if st.session_state.request_count >= 10:  # Max 10 requests per minute
        wait_time = 60 - (current_time - st.session_state.last_request_time)
        if wait_time > 0:
            st.warning(f"⏳ Rate limit protection: Please wait {int(wait_time)} seconds before making another request.")
            return False
    
    st.session_state.request_count += 1
    return True

def generate_course(topic, level, duration):
    if not configure_gemini():
        return None
    
    if not rate_limit_check():
        return None
    
    model_name = get_working_model()
    if not model_name:
        st.error("❌ No working model found. Please check your API key.")
        return None
    
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        st.info(f"🤖 Using model: {model_name}")
            
        prompt = f"""Create a detailed course outline for "{topic}" at {level} level.
        The course should be designed for {duration} of learning.
        
        Format the response as JSON with the following structure:
        {{
            "title": "Course Title",
            "description": "Brief description",
            "modules": [
                {{
                    "name": "Module Name",
                    "topics": ["Topic 1", "Topic 2"],
                    "duration": "Estimated time"
                }}
            ],
            "learning_outcomes": ["Outcome 1", "Outcome 2"],
            "prerequisites": ["Prerequisite 1"]
        }}
        
        Provide only the JSON, no additional text."""
        
        response = model.generate_content(prompt)
        # Clean the response to extract JSON
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        course_data = json.loads(text.strip())
        course_data['created_date'] = datetime.now().strftime("%Y-%m-%d")
        course_data['user_topic'] = topic
        course_data['level'] = level
        return course_data
    except Exception as e:
        st.error(f"Error generating course: {str(e)}")
        return None

def generate_quiz(topic, difficulty, num_questions):
    if not configure_gemini():
        return None
    
    if not rate_limit_check():
        return None
    
    model_name = get_working_model()
    if not model_name:
        st.error("❌ No working model found. Please check your API key.")
        st.info("💡 Get a new API key from: https://aistudio.google.com/app/apikey")
        return None
    
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        st.info(f"🤖 Using model: {model_name}")
            
        prompt = f"""Generate {num_questions} multiple-choice questions about "{topic}" at {difficulty} difficulty level.
        
        Format as JSON:
        {{
            "questions": [
                {{
                    "question": "Question text?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "correct": "A",
                    "explanation": "Why this is correct"
                }}
            ]
        }}
        
        Provide only the JSON."""
        
        response = model.generate_content(prompt)
        # Clean the response to extract JSON
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return None

def get_recommendations(user_history, preferences):
    if not configure_gemini() or not user_history:
        return ["Python Programming", "Data Science Basics", "Web Development", "Machine Learning Introduction"]
    
    model_name = get_working_model()
    if not model_name:
        return ["Python Programming", "Data Science Basics", "Web Development", "Machine Learning Introduction", "AI Fundamentals"]
    
    try:
        model = genai.GenerativeModel(
            model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
            
        prompt = f"""Based on learning history: {user_history} and preferences: {preferences},
        recommend 5 relevant courses. Return only a JSON array of course names:
        ["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]"""
        
        response = model.generate_content(prompt)
        # Clean the response to extract JSON
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        return json.loads(text.strip())
    except:
        return ["Python Programming", "Data Science Basics", "Web Development", "Machine Learning Introduction", "AI Fundamentals"]

# Authentication Page
def auth_page():
    st.markdown('<h1 class="main-header">🎓 AI Learning Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your Personalized Learning Journey Starts Here</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", use_container_width=True):
                if username in st.session_state.users:
                    if st.session_state.users[username] == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        initialize_user_data(username)
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password")
                else:
                    st.error("❌ User not found")
        
        with tab2:
            st.markdown("### Create Your Account")
            new_username = st.text_input("Choose Username", key="signup_username")
            new_password = st.text_input("Choose Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            email = st.text_input("Email Address", key="signup_email")
            
            if st.button("Sign Up", use_container_width=True):
                if new_username and new_password and email:
                    if new_username not in st.session_state.users:
                        if new_password == confirm_password:
                            st.session_state.users[new_username] = hash_password(new_password)
                            st.success("✅ Account created! Please login.")
                        else:
                            st.error("❌ Passwords don't match")
                    else:
                        st.error("❌ Username already exists")
                else:
                    st.error("❌ Please fill all fields")

# Main Dashboard
def dashboard():
    st.markdown(f'<h1 class="main-header">Welcome, {st.session_state.current_user}! 👋</h1>', unsafe_allow_html=True)
    
    # API Key Configuration
    if not st.session_state.api_key:
        st.warning("⚠️ Please configure your Google API Key to use AI features")
        
        st.markdown("""
        ### 📌 Free Tier Limits:
        - **15 requests per minute**
        - **1,500 requests per day**
        - Use **Flash models** for best performance
        
        Get your FREE API key: https://aistudio.google.com/app/apikey
        """)
        
        api_key_input = st.text_input("Enter Google Gemini API Key", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save API Key"):
                st.session_state.api_key = api_key_input
                st.session_state.working_model = None  # Reset working model
                st.success("✅ API Key saved!")
                st.rerun()
        with col2:
            if st.button("🔍 Test API Key"):
                st.session_state.api_key = api_key_input
                available_models, error = test_api_key_and_list_models()
                if available_models:
                    st.success(f"✅ API Key works! Found {len(available_models)} models:")
                    st.write(available_models)
                    # Highlight Flash models
                    flash_models = [m for m in available_models if 'flash' in m.lower()]
                    if flash_models:
                        st.info(f"⚡ Flash models (recommended): {', '.join(flash_models)}")
                else:
                    st.error(f"❌ API Key test failed: {error}")
                    st.info("💡 Get a new key from: https://aistudio.google.com/app/apikey")
        return
    
    user = st.session_state.current_user
    user_stats = st.session_state.user_data[user]
    
    # Stats Dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("📚 Courses", user_stats['courses_completed'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("✅ Quizzes Taken", user_stats['quizzes_taken'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("⭐ Total Score", user_stats['total_score'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-box">', unsafe_allow_html=True)
        st.metric("🔥 Learning Streak", f"{user_stats['learning_streak']} days")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Rate limit info
    if st.session_state.request_count > 0:
        remaining = 10 - st.session_state.request_count
        st.info(f"⚡ API requests this minute: {st.session_state.request_count}/10 (Rate limit protection active)")
    
    st.markdown("---")
    
    # Recommended Courses
    st.markdown('<h2 class="sub-header">🎯 Recommended For You</h2>', unsafe_allow_html=True)
    
    history = [c['user_topic'] for c in user_stats['courses']]
    recommendations = get_recommendations(history, user_stats['learning_preferences'])
    
    cols = st.columns(min(len(recommendations), 3))
    for idx, rec in enumerate(recommendations[:3]):
        with cols[idx]:
            st.markdown(f"""
            <div class="course-card">
                <h3>📘 {rec}</h3>
                <p>Personalized recommendation based on your learning pattern</p>
            </div>
            """, unsafe_allow_html=True)

# Course Generation Page
def course_generator():
    st.markdown('<h1 class="main-header">🎨 Generate Your Course Roadmap</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("📚 What do you want to learn?", placeholder="e.g., Machine Learning, Spanish, Guitar...")
        
        col_a, col_b = st.columns(2)
        with col_a:
            level = st.selectbox("🎓 Skill Level", ["Beginner", "Intermediate", "Advanced"])
        with col_b:
            duration = st.selectbox("⏱️ Duration", ["1 week", "2 weeks", "1 month", "3 months"])
        
        if st.button("🚀 Generate Course", use_container_width=True):
            with st.spinner("Creating your personalized course..."):
                course = generate_course(topic, level, duration)
                
                if course:
                    st.session_state.user_data[st.session_state.current_user]['courses'].append(course)
                    st.session_state.user_data[st.session_state.current_user]['courses_completed'] += 1
                    
                    st.markdown('<div class="success-msg">✅ Course Generated Successfully!</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"### 📖 {course['title']}")
                    st.write(course['description'])
                    
                    st.markdown("#### 📋 Course Modules")
                    for idx, module in enumerate(course['modules'], 1):
                        with st.expander(f"Module {idx}: {module['name']}"):
                            st.write(f"**Duration:** {module['duration']}")
                            st.write("**Topics:**")
                            for topic in module['topics']:
                                st.write(f"- {topic}")
                    
                    st.markdown("#### 🎯 Learning Outcomes")
                    for outcome in course['learning_outcomes']:
                        st.write(f"✓ {outcome}")
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("Be specific about what you want to learn for better course generation!")
        st.markdown("### 📊 Your Courses")
        st.metric("Total Courses", len(st.session_state.user_data[st.session_state.current_user]['courses']))

# Quiz Page
def quiz_page():
    st.markdown('<h1 class="main-header">📝 AI-Powered Quizzes</h1>', unsafe_allow_html=True)
    
    if 'current_quiz' not in st.session_state:
        st.session_state.current_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = None
    
    if st.session_state.current_quiz is None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            quiz_topic = st.text_input("📚 Quiz Topic", placeholder="e.g., Python Basics, World History...")
            
            col_a, col_b = st.columns(2)
            with col_a:
                difficulty = st.selectbox("🎯 Difficulty", ["Easy", "Medium", "Hard"])
            with col_b:
                num_questions = st.slider("❓ Number of Questions", 3, 10, 5)
            
            if st.button("🎲 Generate Quiz", use_container_width=True):
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(quiz_topic, difficulty, num_questions)
                    if quiz:
                        st.session_state.current_quiz = quiz
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_topic = quiz_topic
                        st.rerun()
        
        with col2:
            st.markdown("### 📈 Quiz Stats")
            user_stats = st.session_state.user_data[st.session_state.current_user]
            st.metric("Quizzes Taken", user_stats['quizzes_taken'])
            if user_stats['quizzes_taken'] > 0:
                avg_score = user_stats['total_score'] / user_stats['quizzes_taken']
                st.metric("Average Score", f"{avg_score:.1f}%")
    
    else:
        quiz = st.session_state.current_quiz
        st.markdown(f"### 📝 Quiz: {st.session_state.quiz_topic}")
        
        for idx, q in enumerate(quiz['questions']):
            st.markdown(f'<div class="quiz-card">', unsafe_allow_html=True)
            st.markdown(f"**Question {idx + 1}:** {q['question']}")
            answer = st.radio(f"Select answer for Q{idx + 1}", q['options'], key=f"q_{idx}")
            st.session_state.quiz_answers[idx] = answer[0]
            st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit Quiz", use_container_width=True):
                correct = 0
                for idx, q in enumerate(quiz['questions']):
                    if st.session_state.quiz_answers.get(idx) == q['correct']:
                        correct += 1
                
                score = (correct / len(quiz['questions'])) * 100
                st.session_state.quiz_score = score
                
                user_stats = st.session_state.user_data[st.session_state.current_user]
                user_stats['quizzes_taken'] += 1
                user_stats['total_score'] += score
                user_stats['quiz_history'].append({
                    'topic': st.session_state.quiz_topic,
                    'score': score,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                
                st.balloons()
                st.success(f"🎉 You scored {score:.1f}% ({correct}/{len(quiz['questions'])} correct)")
                
                for idx, q in enumerate(quiz['questions']):
                    user_answer = st.session_state.quiz_answers.get(idx)
                    if user_answer == q['correct']:
                        st.success(f"Q{idx+1}: ✅ Correct!")
                    else:
                        st.error(f"Q{idx+1}: ❌ Incorrect. Correct answer: {q['correct']}")
                        st.info(f"💡 {q['explanation']}")
        
        with col2:
            if st.button("🔄 New Quiz", use_container_width=True):
                st.session_state.current_quiz = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = None
                st.rerun()

# Analytics Page
def analytics_page():
    st.markdown('<h1 class="main-header">📊 Learning Analytics</h1>', unsafe_allow_html=True)
    
    user_stats = st.session_state.user_data[st.session_state.current_user]
    
    if not user_stats['quiz_history']:
        st.info("📚 Take some quizzes to see your analytics!")
        return
    
    # Performance over time
    df = pd.DataFrame(user_stats['quiz_history'])
    df['date'] = pd.to_datetime(df['date'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(df, x='date', y='score', title='📈 Quiz Performance Over Time',
                      labels={'score': 'Score (%)', 'date': 'Date'})
        fig.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        topic_scores = df.groupby('topic')['score'].mean().reset_index()
        fig = px.bar(topic_scores, x='topic', y='score', title='📚 Average Score by Topic',
                     labels={'score': 'Average Score (%)', 'topic': 'Topic'})
        fig.update_traces(marker_color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)
    
    # Progress metrics
    st.markdown("### 🎯 Progress Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_score = df['score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}%", f"+{avg_score-50:.1f}% from baseline")
    
    with col2:
        improvement = df['score'].iloc[-1] - df['score'].iloc[0] if len(df) > 1 else 0
        st.metric("Improvement", f"{improvement:+.1f}%", "Since first quiz")
    
    with col3:
        best_score = df['score'].max()
        st.metric("Best Score", f"{best_score:.1f}%", "Personal best")

# Profile Page
def profile_page():
    st.markdown('<h1 class="main-header">👤 My Profile</h1>', unsafe_allow_html=True)
    
    user = st.session_state.current_user
    user_stats = st.session_state.user_data[user]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 👨‍🎓 User Info")
        st.write(f"**Username:** {user}")
        st.write(f"**Member Since:** {datetime.now().strftime('%B %Y')}")
        st.write(f"**Status:** Active Learner")
    
    with col2:
        st.markdown("### 🎯 Learning Preferences")
        prefs = st.multiselect(
            "Select your interests",
            ["Programming", "Data Science", "AI/ML", "Web Dev", "Mobile Dev", 
             "Cloud Computing", "Cybersecurity", "Languages", "Business", "Design"],
            default=user_stats.get('learning_preferences', [])
        )
        if st.button("💾 Save Preferences"):
            user_stats['learning_preferences'] = prefs
            st.success("Preferences saved!")
    
    st.markdown("---")
    st.markdown("### 📚 My Courses")
    
    if user_stats['courses']:
        for idx, course in enumerate(user_stats['courses']):
            with st.expander(f"📖 {course['title']}"):
                st.write(f"**Level:** {course['level']}")
                st.write(f"**Created:** {course['created_date']}")
                st.write(f"**Modules:** {len(course['modules'])}")
    else:
        st.info("No courses yet. Generate your first course!")

# Main App Logic
def main():
    if not st.session_state.logged_in:
        auth_page()
    else:
        # Sidebar
        with st.sidebar:
            st.markdown(f"### 👋 {st.session_state.current_user}")
            st.markdown("---")
            
            page = st.radio(
                "Navigation",
                ["🏠 Dashboard", "🎨 Generate Course", "📝 Take Quiz", "📊 Analytics", "👤 Profile"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.rerun()
        
        # Route to pages
        if "Dashboard" in page:
            dashboard()
        elif "Generate Course" in page:
            course_generator()
        elif "Take Quiz" in page:
            quiz_page()
        elif "Analytics" in page:
            analytics_page()
        elif "Profile" in page:
            profile_page()

if __name__ == "__main__":
    main()