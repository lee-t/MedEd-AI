import sys

if "pyodide" in sys.modules:
    # psutil doesn't work on pyodide--use fake data instead
    from fakepsutil import cpu_count, cpu_percent
else:
    from psutil import cpu_count, cpu_percent

import matplotlib
import numpy as np
import pandas as pd
from helpers import  check_new_data, accuracy, completeness, average_accuracy_per_question_bar, distribution_of_scores_hist, instructor_feedback, get_courses, get_quizzes
from shiny.express import input, output, render, ui
from shiny import reactive
from shinywidgets import output_widget, render_widget 
from dotenv import load_dotenv
import os

# The agg matplotlib backend seems to be a little more efficient than the default when
# running on macOS, and also gives more consistent results across operating systems
matplotlib.use("agg")

# max number of samples to retain
MAX_SAMPLES = 1000
# secs between samples
SAMPLE_PERIOD = 1

ncpu = cpu_count(logical=True)

ui.page_opts(fillable=True)

load_dotenv('/Credentials/.env')
api_key = os.getenv("CANVAS_API_KEY")

ui.tags.style(
    """
    /* Don't apply fade effect, it's constantly recalculating */
    .recalculating, .recalculating > * {
        opacity: 1 !important;
    }
    """
)

ui.busy_indicators.use(spinners=False, pulse=True)

with ui.sidebar():
    ui.input_password("apikey", "API Key:", api_key)
    ui.input_action_button("go", "Go"),

    ui.input_selectize(
        "course",
        "Choose Course",
        choices={
            "test": "",}
    )


    ui.input_action_button("next", "Next"),

    ui.input_selectize(
        "cae",
        "Choose CAE",
        choices={
            "test": "",}
    )


    ui.input_action_button("generate", "Generate Reports"),

    #API key entry to fetch all courses
    @reactive.effect
    @reactive.event(input.go)
    def _():
        if input.apikey() =="":
            print("Please Enter a valid API Key")
        else:
            courses_dict = get_courses(input.apikey())
            ui.update_selectize("course", choices=courses_dict)

    #Choose the course
    @reactive.effect
    @reactive.event(input.next)
    def _():
        if input.course():
            courses_dict = get_courses(input.apikey())
            quiz_dict = get_quizzes(input.apikey(), input.course())
            ui.update_selectize("cae", choices=quiz_dict)            
        else:
            print("No course input")

with ui.panel_absolute(width="75%"):
    # Enable busy indicators
    with ui.navset_bar(title="Student Performance"):
        @reactive.event(input.generate)
        def _():
            check_new_data(input.course(), input.cae())     

        with ui.nav_panel(title="Graphs"):
            #avg accuracy per question plot
            @render_widget
            @reactive.event(input.generate)
            def plot_average_accuracy_per_question_bar():
                return average_accuracy_per_question_bar(input.course(), input.cae())

            #dist of accuracy&completeness scores histogram
            @render_widget
            @reactive.event(input.generate)
            def plot_distribution_of_scores_hist():
                return distribution_of_scores_hist(input.course(), input.cae())

            #accuracy plot across similar questions
            @render_widget
            @reactive.event(input.generate)
            def plot_accuracy():
                return accuracy(input.course(), input.cae())

            #completeness plot across similar questions
            @render_widget
            @reactive.event(input.generate)
            def plot_completeness():
                return completeness(input.course(), input.cae())


        with ui.nav_panel(title="Topics"):
            ui.input_numeric("table_rows", "#Under construction", 0)
            @render.text
            @reactive.event(input.generate)
            def feedback():
                return instructor_feedback(input.course(), input.cae())

        with ui.nav_panel(title="Source Data"):
            @render.download(label="Download CSV", filename="data.csv")
            @reactive.event(input.download)
            def _():
                df = pd.read_json('/Data/graded_quizzes_202407111420.json')
                subset = df[(df['quiz_id']==int(input.cae())) & (df['course_id']==int(input.course()))]
                yield subset.to_csv()

            @render.data_frame
            @reactive.event(input.generate)
            def table():
                df = pd.read_json('/Data/graded_quizzes_202407111420.json')
                subset = df[(df['quiz_id']==int(input.cae())) & (df['course_id']==int(input.course()))]
                return render.DataGrid(subset)
