from dotenv import load_dotenv
import json
from os import getenv
from requests import get

load_dotenv()
headers = {"Authorization": f"Bearer {getenv('ALBERT_PERSONAL_TOKEN')}"}

BASE = "https://api-inside.albertschool.com"
USER_PROFILE_PATH = "/user/user-profile"
PROGRAMS_PATH = "/public/programs"
PROGRAM_EXAMPLE_PATH = "/public/program-versions/{PROGRAM_VERSION_ID}"
SYLLABUS_EXAMPLE_PATH = "/public/program-versions/{PROGRAM_ID}/course-module-versions/{COURSE_MODULE_VERSION_ID}"
COURSES_PATH = f"/student/{getenv('ALBERT_USER_ID')}/course-module-instances"
COURSE_EXAMPLE_PATH = "/course/course-module-instance/by-id/{COURSE_ID}"
COURSE_EXAMPLE_DOCUMENTS_PATH = "/course/academic-documents/by-course-module-instance/{COURSE_ID}?page=1&limit=50&include_archived=false&include_sessions=true"
ATTENDANCE_PATH = f"/attendance/user/{getenv('ALBERT_USER_ID')}"
GRADES_PATH = f"/student-exam-grade/student/{getenv('ALBERT_STUDENT_ID')}"


def get_content(path: str) -> dict | list:
    url = BASE + path
    try:
        r = get(url, headers=headers)
        return r.json()
    except Exception as e:
        print(path, "ERROR", e)


content = {
    "user_profile": get_content(USER_PROFILE_PATH),
    "programs": get_content(PROGRAMS_PATH),
    "program_example": get_content(
        PROGRAM_EXAMPLE_PATH.format(
            PROGRAM_VERSION_ID="154234bf-15a3-42e6-b2f5-e591ac0d36aa"
        )
    ),  # "Bachelor of Business and Data"
    "syllabus_example": get_content(
        SYLLABUS_EXAMPLE_PATH.format(
            PROGRAM_ID="154234bf-15a3-42e6-b2f5-e591ac0d36aa",
            COURSE_MODULE_VERSION_ID=81,
        )
    ),  # "Bachelor of Business and Data" --> "Introduction to generative AI"
    "courses": get_content(COURSES_PATH),
    "course_example": get_content(
        COURSE_EXAMPLE_PATH.format(COURSE_ID=1508)
    ),  # "Introduction to generative AI"
    "course_example_documents": get_content(
        COURSE_EXAMPLE_DOCUMENTS_PATH.format(COURSE_ID=1508)
    ),  # "Introduction to generative AI"
    "attendance": get_content(ATTENDANCE_PATH),
    "grades": get_content(GRADES_PATH),
}

with open("inside-albert/api_content.json", "w") as f:
    json.dump(content, f, indent=4, ensure_ascii=False)
