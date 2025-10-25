from pydantic import BaseModel
import anthropic
import json
import os
from typing import Optional, List, Union

with open('api_key.json') as f:
    api_keys = json.load(f)
    antropic_key = api_keys['antropic_key']
    grok_key = api_keys['grok_key']

from groq import Groq

class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY") if os.environ.get("GROQ_API_KEY") else grok_key
        )

    def generate_report(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional report writer specializing in mental health analysis. Format your response in clear sections with headers. Focus on analyzing the data based on the prediction values (0 or 1) indicating mental health issues."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant"
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"

class AnthropicLanguageModel:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=antropic_key
        )

    def generate_report(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                system="You are a professional report writer specializing in mental health analysis. Format your response in clear sections with headers.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.content[0].text

        except Exception as e:
            print(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"

class QuestionnaireDataModel(BaseModel):
    answers: List[dict]
    source: str
    
class QuestionnaireColumnsModel(BaseModel):
    diet: str
    ethnic_group: str
    hours_per_week_university_work: int
    family_earning_class: str
    quality_of_life: str
    alcohol_consumption: str
    personality_type: str
    stress_in_general: List[str]
    well_hydrated: str
    exercise_per_week: int
    known_disabilities: str
    work_hours_per_week: int
    financial_support: str
    form_of_employment: str
    financial_problems: str
    home_country: str
    age: int
    course_of_study: str
    stress_before_exams: str
    feel_afraid: str
    timetable_preference: str
    timetable_reasons: str
    timetable_impact: str
    total_device_hours: int
    hours_socialmedia: int
    level_of_study: str
    gender: str
    physical_activities: str
    hours_between_lectures: int
    hours_per_week_lectures: int
    hours_socialising: int
    actual: str
    student_type_time: str
    student_type_location: str
    cost_of_study: int
    sense_of_belonging: str
    mental_health_activities: str
    source: Optional[str]
    predictions: Optional[int]
    captured_at: Optional[str]
    
class DashboardDataModel(BaseModel):
    diet: Optional[str]
    ethnic_group: Optional[str]
    hours_per_week_university_work: Optional[int]
    family_earning_class: Optional[str]
    quality_of_life: Optional[str]
    alcohol_consumption: Optional[str]
    personality_type: Optional[str]
    stress_in_general: Optional[str]
    well_hydrated: Optional[str]
    exercise_per_week: Optional[int]
    known_disabilities: Optional[str]
    work_hours_per_week: Optional[int]
    financial_support: Optional[str]
    form_of_employment: Optional[str]
    financial_problems: Optional[str]
    home_country: Optional[str]
    age: Optional[int]
    course_of_study: Optional[str]
    stress_before_exams: Optional[str]
    feel_afraid: Optional[str]
    timetable_preference: Optional[str]
    timetable_reasons: Optional[str]
    timetable_impact: Optional[str]
    total_device_hours: Optional[int]
    hours_socialmedia: Optional[int]
    level_of_study: Optional[str]
    gender: Optional[str]
    physical_activities: Optional[str]
    hours_between_lectures: Optional[int]
    hours_per_week_lectures: Optional[int]
    hours_socialising: Optional[int]
    actual: Optional[str]
    student_type_time: Optional[str]
    student_type_location: Optional[str]
    cost_of_study: Optional[int]
    sense_of_belonging: Optional[str]
    mental_health_activities: Optional[str]
    source: Optional[str]
    predictions: Optional[int]
    captured_at: Optional[str]
    
    
GoogleFormsTranslationMap = {
    "1. Would you describe your current diet as healthy and balanced?": "diet",
    "2. What is your ethnic group?": "ethnic_group",
    "3. How many hours do you spend on university/academic-related work, separate from your Course Timetable, per week during exams?": "hours_per_week_university_work",
    "4. How would you rate your family class? (family earnings per year)": "family_earning_class",
    "5. How would you define your quality of life? (as defined by the World Health Organization)": "quality_of_life",
    "6. How would you define your alcohol consumption?": "alcohol_consumption",
    "7. Would you consider yourself an introvert or extrovert person? (Definitions from Oxford Languages)": "personality_type",
    "8. In general, do you feel you experience stress while in the University/Academic Studies? (tick all that apply)": "stress_in_general",
    "9. Would you say that you are normally well hydrated?": "well_hydrated",
    "10. How often do you exercise per week?": "exercise_per_week",
    "11. Do you have any known disabilities?": "known_disabilities",
    "12. How many hours per week do you work?": "work_hours_per_week",
    "13. What is your main financial support for your studies?": "financial_support",
    "14. Are you in any form of employment?": "form_of_employment",
    "15. Do you normally encounter financial issues paying your fees?": "financial_problems",
    "16. What Country are you from?": "home_country",
    "17. What is your year of birth?": "age",  # przeliczane z roku
    "18. What is your course of study?": "course_of_study",
    "19. Do you normally feel stressed before exams?": "stress_before_exams",
    "20. How often in the last week or two did you feel afraid that something awful might happen?": "feel_afraid",
    "21. If your Course has less than 24 hours on Timetable, would you prefer your timetable to be spread (3-4 days with fewer lectures) or compact (1-2 busy days) so you have less stress at university? (eg, 1-2 busy days or 3-4 days with less lectures)": "timetable_preference",
    "22. What are the reasons for your timetable preference?": "timetable_reasons",
    "23. Do you feel your timetabling structure has any impact on your study, life and health?": "timetable_impact",
    "24. How many hours do you spend using technology devices per day (mobile, desktop, laptops, etc)?": "total_device_hours",
    "25. How many hours do you spend using social media per day (Instagram, Tiktok, Twitter, etc)?": "hours_socialmedia",
    "26. What year of study are you in?": "level_of_study",
    "27. How would you describe your biological gender?": "gender",
    "28. Do you consider physical activity to be helpful to your mental health?": "physical_activities",
    "29. How many hours do you normally have BETWEEN lectures per day?": "hours_between_lectures",
    "30. How many hours per week do you have active lectures? (Active means attending lectures)": "hours_per_week_lectures",
    "31. How many hours per week do you socialise? (Meeting other people, participating in social activities etc).": "hours_socialising",
    "32. Would you classify yourself or have you been diagnosed with mental health issues by a professional?": "actual",
    "33. Are you full-time or part-time student?": "student_type_time",
    "34. Are you a home student or an international student?": "student_type_location",
    "35. What are the approximate costs for your studies? (tuition fee per year of study, in pound sterling £)": "cost_of_study",
    "36. Do you feel a sense of \"belonging\" at UAL?": "sense_of_belonging",
    "37. Please let us know about any activities that could support your mental health that you would be interested in. (e.g., physical activities, sports, mindfulness, book clubs, arts/craft activities etc)": "mental_health_activities",
}

QuestionNumberToField = {
    1: "diet",
    2: "ethnic_group",
    3: "hours_per_week_university_work",
    4: "family_earning_class",
    5: "quality_of_life",
    6: "alcohol_consumption",
    7: "personality_type",
    8: "stress_in_general",
    9: "well_hydrated",
    10: "exercise_per_week",
    11: "known_disabilities",
    12: "work_hours_per_week",
    13: "financial_support",
    14: "form_of_employment",
    15: "financial_problems",
    16: "home_country",
    17: "age",
    18: "course_of_study",
    19: "stress_before_exams",
    20: "feel_afraid",
    21: "timetable_preference",
    22: "timetable_reasons",
    23: "timetable_impact",
    24: "total_device_hours",
    25: "hours_socialmedia",
    26: "level_of_study",
    27: "gender",
    28: "physical_activities",
    29: "hours_between_lectures",
    30: "hours_per_week_lectures",
    31: "hours_socialising",
    32: "actual",
    33: "student_type_time",
    34: "student_type_location",
    35: "cost_of_study",
    36: "sense_of_belonging",
    37: "mental_health_activities",
}
