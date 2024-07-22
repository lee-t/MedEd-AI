from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import requests
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as AzureOAI
from llama_index.core import Settings
from RAG import create_sql_engine, create_query_engine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SQLAutoVectorQueryEngine
import re
import plotly.express as px
import plotly.graph_objects as go

load_dotenv('/Credentials/.env')

#Azure OpenAI Creds
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
credential = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = "2024-04-01-preview"
azure_openai_embedding_deployment = "text-embedding-ada-002"
embedding_model_name = "text-embedding-ada-002"
llm_model_name = "gpt-4o"
api_type = "azure"
api_key = os.getenv("CANVAS_API_KEY")

llm = AzureOAI(
            model = llm_model_name,
            deployment_name = llm_model_name,
            api_key = credential,
            azure_endpoint = endpoint,
            api_version = azure_openai_api_version,
            api_type = api_type
        )

embed_model = AzureOpenAIEmbedding(
            model = embedding_model_name,
            deployment_name = embedding_model_name,
            api_key = credential,
            azure_endpoint = endpoint,
            api_version = azure_openai_api_version,
            api_type = api_type,
            embed_batch_size=50
        )

Settings.llm = llm
Settings.embed_model = embed_model

def get_courses(api_key):
    url = "https://canvas.harvard.edu/api/v1/courses"
    headers = {"Authorization": f"Bearer {api_key}"}
    courses = []
    courses_dict={}
    page = 1
    while True:
        params = {"page": page}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
            courses_page = json.loads(response.text)  # Convert response text to dictionary using json.loads()
            for course in courses_page:
                courses_dict[str(course['id'])] = course['name']
            if "next" not in response.links:
                break  # No more pages to fetch
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching courses")
    return courses_dict

def get_quizzes(api_key, course):
    quizzes = []
    quiz_dict = {}
    quiz_dict[str(course)] = {}
    url = f"https://canvas.harvard.edu/api/v1/courses/{course}/quizzes"
    headers = {"Authorization": f"Bearer {api_key}"}
    page = 1
    while True:
        params = {"page": page, "per_page":"100"}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
            quizzes_page = json.loads(response.text)  # Convert response text to dictionary using json.loads()
            for quiz in quizzes_page:
                if quiz['html_url'].replace('https://canvas.harvard.edu/courses/','')[0:6] == course:
                    if 'Consolidation' in quiz['title']:
                        quiz_dict[str(course)][str(quiz['id'])] = quiz['title']
                else:
                    pass
            if "next" not in response.links:
                break  # No more pages to fetch
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching quizzes for course {course}: {e}")
        
        sorted_quiz_dict = dict(sorted(quiz_dict[course].items(), key=lambda item: item[1]))

    return sorted_quiz_dict


# Load JSON data into a DataFrame
def load_json_to_dataframe(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    newdf = pd.DataFrame(data)
    return newdf

def extract_text(html):
    # Regex pattern to match the text between the tags
    pattern = re.compile(r'>\s*([^<]+?)\s*<')

    # Find all matches
    matches = pattern.findall(html)

    # Join the matches to form the complete extracted text
    extracted_text = ' '.join(matches).strip()

    return extracted_text

# Function to compare two dataframes and find differences
def check_new_data(course, quiz):
    headers = {"Authorization": f"Bearer {api_key}"}
    pre_graded_df = pd.read_json('/Data/graded_quizzes_202407111420.json')
    un_graded = []

    try:
        url = f"https://canvas.harvard.edu/api/v1/courses/{course}/quizzes/{quiz}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        quiz_page = json.loads(response.text)  # Convert response text to dictionary using json.loads()
        assignment_id = quiz_page['assignment_id']

        url = f"https://canvas.harvard.edu/api/v1/courses/{course}/quizzes/{quiz}/questions"
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        questions_page = json.loads(response.text)  # Convert response text to dictionary using json.loads()
        questions_df = pd.DataFrame.from_dict(questions_page)
        questions_df = questions_df[(questions_df['question_type']=="essay_question")]

        page = 1
        while True:
            url = f"https://canvas.harvard.edu/api/v1/courses/{course}/assignments/{assignment_id}/submissions"
            params = {"page": page, "include[]":"submission_history","per_page":"100"}

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
            submissions_page = json.loads(response.text)  # Convert response text to dictionary using json.loads()
            for user_submission in submissions_page:
                submission_id = user_submission['id']
                student_score = user_submission['score']
                attempt = user_submission['attempt']

                check_if_graded = pre_graded_df[(pre_graded_df['submission_id']==submission_id)]

                if user_submission['submission_history'][0].get('submission_data'):
                    for submission_data in user_submission['submission_history'][0].get('submission_data'):
                        write_dict = {'quiz_id':'', 
                                    'quiz_type':'', 
                                    'quiz_title':'',
                                    'history_id':'',
                                    'submission_id':'',
                                    'student_score':'',
                                    'quiz_question_count':'',
                                    'quiz_points_possible':'',
                                    'question_points_possible':'',
                                    'answer_points_scored':'',
                                    'attempt':'',
                                    'question_name':'',
                                    'question_type':'',
                                    'question_text':'',
                                    'question_answer':'',
                                    'student_answer':'',
                                    'course_id':'',
                                    'accuracy':'',
                                    'completeness':''}

                        if check_if_graded.empty:
                            #from quiz_page
                            write_dict['quiz_id'] = quiz
                            write_dict['quiz_type'] = quiz_page['quiz_type']
                            write_dict['quiz_title'] = quiz_page['title']
                            write_dict['quiz_question_count'] = quiz_page['question_count']
                            write_dict['quiz_points_possible'] = quiz_page['points_possible']
                            write_dict['question_points_possible'] = quiz_page['points_possible']
        
                            #from questions_df
                            questions_df_filtered = questions_df[(questions_df['id']==submission_data['question_id']) 
                                                                & (questions_df['quiz_id']==int(quiz))]
                            
                            write_dict['question_text'] = extract_text(questions_df_filtered['question_text'].item())
                            write_dict['question_name'] = questions_df_filtered['question_name'].item()
                            write_dict['question_type'] = questions_df_filtered['question_type'].item()
                            write_dict['question_answer'] = questions_df_filtered['neutral_comments'].item()                 

                            #from submission_data
                            write_dict['history_id'] = submission_data['question_id']
                            write_dict['answer_points_scored'] = submission_data['points']
                            write_dict['student_answer'] = submission_data['text']
                            write_dict['attempt'] = attempt
                            write_dict['submission_id'] = submission_id
                            write_dict['student_score'] = student_score
                            write_dict['course_id'] = course

                            accuracy, completeness = grade_answer(write_dict['student_answer'], write_dict['question_answer'])
                            write_dict['accuracy'] = accuracy
                            write_dict['completeness'] = completeness
                            
                            #reorder the columns
                            write_dict = pd.DataFrame.from_records(write_dict, index=[0])
                            write_dict = write_dict[pre_graded_df.columns]
                            write_dict = write_dict.to_dict('records')[0]
                            
                            un_graded.append(write_dict)

                        else:
                            pass

            if "next" not in response.links:
                break  # No more pages to fetch
            page += 1
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching quizzes for course {course}: {e}")
        
    if len(un_graded) > 0:
        with open(f'/Data/graded_quizzes_202407111420.json', "r") as file:
            data = file.read()[:-1]
        with open(f'/Data/graded_quizzes_202407111420.json', "w") as file:
            file.write(data)
            file.write(',')
        with open(f'/Data/graded_quizzes_202407111420.json', "a") as outfile:
            for record in un_graded:
                if record == un_graded[-1]:
                    json.dump(record, outfile, indent=2)                
                else:
                    json.dump(record, outfile, indent=2)
                    outfile.write(',')
                    outfile.write('\n')
            outfile.write(']')

def grade_answer(student_answer, correct_answer):
    prompt = "Compare the student answer to the correct answer. Rate the accuracy (a measure of how correct the student is) and completeness (did the student identify all components of the question) of the student answer according to these scales: Accuracy Options: 1 - not accurate, 2 - somewhat accurate, 3 - mostly accurate, 4 - completely accurate. Completeness: 1 - incomplete, 2 - partially complete, 3 - mostly complete, 4 - complete. Explain your answer briefly. Format your answer as a list separated by |. Example: 3|4|explanation" +f"Student Answer:{student_answer}\nCorrect Answer:{correct_answer}."

    client = AzureOpenAI(
            api_key = credential,
            azure_endpoint = endpoint,
            api_version = azure_openai_api_version
        )

    response = client.chat.completions.create(  model = llm_model_name,
                                                messages=[
                                                    {"role": "system", "content": "You are a helpful course Teaching Assistant."},
                                                    {"role": "user", "content": f"{prompt}"}
                                                ]
                                            )

    grade = response.choices[0].message.content.split('|')

    accuracy = grade[0]
    completeness = grade[1]

    return accuracy, completeness


def instructor_feedback(course, quiz): 

#TODO: ideas for instructor feedback:
#Hierarchical
#Do the same as the grading, turn it into a prompt
#Grey's Idea
#Sort the student scores so you have all the poor performing ones on one end
#and the good performing ones are on the other. 
#Level one: why did the ones get ones? twos get twos?
#Level two: Concepts missed/correct per group
#Level three: Overall concepts missed/correct
#Level four: Report
#Add a word limit in 500 words or less in the prompt


    sql_query_engine = create_sql_engine()
    retriever_query_engine = create_query_engine()

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        description=(
            "Useful for translating a natural language query into a SQL query over"
            "a table graded_quizzes, containing columns:"
            "quiz_id (INTEGER), quiz_type (VARCHAR), quiz_title (VARCHAR), history_id (BIGINT), submission_id (BIGINT),"
            "student_score (DOUBLE PRECISION), quiz_question_count (BIGINT), quiz_points_possible (DOUBLE PRECISION), question_points_possible (DOUBLE PRECISION),"
            "answer_points_scored (DOUBLE PRECISION), attempt (BIGINT), question_name (VARCHAR), question_type (VARCHAR), question_text (VARCHAR), question_answer (VARCHAR), student_answer (VARCHAR),"
            "course_id (VARCHAR), accuracy (INTEGER), completeness (INTEGER)"),
    )


    vector_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine,
            description=f"Useful for answering semantic questions about consolidation assessments, and general course-related questions like when certain material is being taught",
        )


    query_engine = SQLAutoVectorQueryEngine(
    sql_tool, 
    vector_tool,
    llm=llm
    ) 

    response = query_engine.query(f"List all correct answers to Question 1 question answer for quiz_id '{int(quiz)}'.")

    # response = query_engine.query(f"For Question 1 of course_id '{int(course)}' and quiz_id '{int(quiz)}', compare student answers to the question answer. Do not include the table name in your SQL statement. What concept did students best understand? Which concept was most frequently not mentioned?")
    # response = query_engine.query(f"For course_id {int(course)} and quiz_id {int(quiz)}, which questions did the students have the worst average completeness and average accuracy?")
    return response.response
                

def plot_distribution(course, quiz):  

    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id']==int(quiz)) & (df['course_id']==int(course))]

    # Assuming 'subset' is your pre-filtered DataFrame
    # Plot distribution of accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(subset['accuracy'], kde=True)
    plt.title('Distribution of Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')

    # Plot distribution of completeness
    plt.subplot(1, 2, 2)
    sns.histplot(subset['completeness'], kde=True)
    plt.title('Distribution of Completeness')
    plt.xlabel('Completeness')
    plt.ylabel('Frequency')

    plt.tight_layout()
    
    return plt.gcf()

def plot_question_performance(course, quiz):
    
    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id']==int(quiz)) & (df['course_id']==int(course))]
        
    question_performance = subset.groupby('question_name')[['accuracy', 'completeness']].mean().reset_index()

    # Bar plot of average accuracy per question
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(data=question_performance, x='accuracy', y='question_name', palette='viridis')
    plt.title('Average Accuracy per Question')
    plt.xlabel('Average Accuracy')
    plt.ylabel('Question Name')

    # Bar plot of average completeness per question
    plt.subplot(1, 2, 2)
    sns.barplot(data=question_performance, x='completeness', y='question_name', palette='viridis')
    plt.title('Average Completeness per Question')
    plt.xlabel('Average Completeness')
    plt.ylabel('Question Name')

    plt.tight_layout()
    return plt.gcf()

#Distribution of accuracy/completeness histogram, not stacked

def accuracy(course, quiz):
#TODO: rather than accuracy across questions, do average accuracy per quiz, from start of course to end
    # Load the JSON data
    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id'] == int(quiz)) & (df['course_id'] == int(course))]
    quiz_group = subset['quiz_title'].unique().item().split(' ')[0]
    
    # avg = subset.groupby('submission_id', as_index=False)['accuracy'].mean()

    # Filter for quizzes with 'Histology' in the title
    quiz_group_subset = df[df['quiz_title'].str.startswith(quiz_group)]

    # Calculate average accuracy per question per quiz
    average_accuracy_per_question = quiz_group_subset.groupby(['quiz_title', 'question_name'])['accuracy'].mean().reset_index()

    # Round the averages to the nearest hundredth
    average_accuracy_per_question['accuracy'] = average_accuracy_per_question['accuracy'].round(2)

    # Create a plotly figure
    fig = go.Figure()

    # Get unique quizzes
    quizzes = average_accuracy_per_question['quiz_title'].unique()

    # Plot each quiz separately
    for quiz in quizzes:
        quiz_data = average_accuracy_per_question[average_accuracy_per_question['quiz_title'] == quiz]
        fig.add_trace(go.Scatter(
            x=quiz_data['question_name'], 
            y=quiz_data['accuracy'], 
            mode='lines+markers', 
            name=quiz,
            hoverinfo='text',
            text=quiz_data['accuracy']
        ))

    # Update layout
    fig.update_layout(
        title=f'Average Accuracy Per Question for {quiz_group} Quizzes',
        xaxis_title='Question Name',
        yaxis_title='Average Accuracy',
        legend_title='Quiz Title',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def completeness(course, quiz):

    # Load the JSON data
    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id'] == int(quiz)) & (df['course_id'] == int(course))]
    quiz_group = subset['quiz_title'].unique().item().split(' ')[0]

    # Filter for quizzes with 'Histology' in the title
    quiz_group_subset = df[df['quiz_title'].str.startswith(quiz_group)]

    # Calculate average completeness per question per quiz
    average_completeness_per_question = quiz_group_subset.groupby(['quiz_title', 'question_name'])['completeness'].mean().reset_index()

    # Round the averages to the nearest hundredth
    average_completeness_per_question['completeness'] = average_completeness_per_question['completeness'].round(2)

    # Create a plotly figure
    fig = go.Figure()

    # Get unique quizzes
    quizzes = average_completeness_per_question['quiz_title'].unique()

    # Plot each quiz separately
    for quiz in quizzes:
        quiz_data = average_completeness_per_question[average_completeness_per_question['quiz_title'] == quiz]
        fig.add_trace(go.Scatter(
            x=quiz_data['question_name'], 
            y=quiz_data['completeness'], 
            mode='lines+markers', 
            name=quiz,
            hoverinfo='text',
            text=quiz_data['completeness']
        ))

    # Update layout
    fig.update_layout(
        title=f'Average Completeness Per Question for {quiz_group} Quizzes',
        xaxis_title='Question Name',
        yaxis_title='Average Completeness',
        legend_title='Quiz Title',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def average_accuracy_per_question_bar(course, quiz):
    # Load the JSON data
    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id'] == int(quiz)) & (df['course_id'] == int(course))]

    quiz_group = subset['quiz_title'].unique().item().split(' ')[0]

    average_accuracy = subset.groupby('question_name')['accuracy'].mean().reset_index()
    average_accuracy['accuracy'] = average_accuracy['accuracy'].round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=average_accuracy['question_name'],
        y=average_accuracy['accuracy'],
        text=average_accuracy['accuracy'],
        textposition='auto',
        hoverinfo='none'  # Disable hover data
    ))

    fig.update_layout(
        title= f'Average Accuracy per Question - {quiz_group}',
        xaxis_title='Question Name',
        yaxis_title='Average Accuracy',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def distribution_of_scores_hist(course, quiz):
    # Load the JSON data
    df = pd.read_json('/Data/graded_quizzes_202407111420.json')

    # Filter the DataFrame
    subset = df[(df['quiz_id'] == int(quiz)) & (df['course_id'] == int(course))]

    quiz_group = subset['quiz_title'].unique().item().split(' ')[0]
    
    # Calculate average accuracy and completeness per question
    average_metrics = subset.groupby('question_name')[['accuracy', 'completeness']].mean().reset_index()
    average_metrics['accuracy'] = average_metrics['accuracy'].round(2)
    average_metrics['completeness'] = average_metrics['completeness'].round(2)

    fig = go.Figure()

    # Add accuracy bars
    fig.add_trace(go.Bar(
        x=average_metrics['question_name'],
        y=average_metrics['accuracy'],
        name='Accuracy',
        text=average_metrics['accuracy'],
        textposition='auto'
    ))

    # Add completeness bars
    fig.add_trace(go.Bar(
        x=average_metrics['question_name'],
        y=average_metrics['completeness'],
        name='Completeness',
        text=average_metrics['completeness'],
        textposition='auto'
    ))

    fig.update_layout(
        title=f'Average Accuracy and Completeness per Question - {quiz_group}',
        xaxis_title='Question Name',
        yaxis_title='Average Score',
        barmode='group',  # Group bars next to each other
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig