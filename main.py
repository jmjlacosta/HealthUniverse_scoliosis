from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware  # <- ADD THIS LINE
from typing import Annotated, Literal
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import io
import uuid
import os

app = FastAPI(
    title="Scoliosis Detection Tool",
    description="""The Posture and Scoliosis Tool is a FastAPI application that evaluates a patient's 
    risk for posture-related issues and scoliosis based on both a structured questionnaire and a side-view 
    image. It processes survey inputs covering demographics, symptoms, lifestyle, and physical indicators, 
    and analyzes an uploaded image using Mediapipe to identify body landmarks and calculate the hip angle. 
    Based on this information, it assigns a risk score (Low, Moderate, or High) and returns an annotated 
    version of the image with pose tracings. The app is suitable for use in health screening workflows, 
    telehealth platforms, or wellness assessments.""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/processed_images", StaticFiles(directory="processed_images"), name="processed_images")

class AssessmentOutput(BaseModel):
    risk_level: str
    score: int
    hip_angle: float | None
    processed_image_link: str

@app.post("/assess", response_model=AssessmentOutput)
def assess_posture(
    request: Request,
    age: Annotated[str, Form(description="What is the patient's age?")] = "20",
    height: Annotated[str, Form(description="What is the patient's height? (cm)")] = "170",
    weight: Annotated[str, Form(description="What is the patient's weight? (kg)")] = "70",
    gender: Annotated[Literal["Male", "Female", "Prefer not to say", "Other"], Form(description="What is the patient's gender?")] = "Prefer not to say",
    occupation: Annotated[Literal["Student", "Office Worker", "Manual Labor", "Other"], Form(description="What is the patient's primary occupation?")] = "Other",
    pain_present: Annotated[Literal["Yes", "No"], Form(description="Does the patient experience any pain in their back or neck?")] = "No",
    pain_location: Annotated[Literal["Upper Back", "Lower Back", "Neck", "Shoulders", "Other"], Form(description="Where does the patient primarily feel the pain?")] = "",
    pain_severity: Annotated[Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], Form(description="On a scale of 1 to 10, how would the patient rate their pain?")] = "0",
    pain_duration: Annotated[Literal["Less than a month", "1-6 months", "6 months to a year", "Over a year"], Form(description="How long has the patient been experiencing this pain?")] = "",
    symptom_onset: Annotated[Literal["Gradually", "Suddenly"], Form(description="Did the patient's pain start gradually or suddenly?")] = "",
    activity_related_pain: Annotated[Literal["Yes", "No"], Form(description="Does physical activity worsen the patient's pain?")] = "",
    previous_diagnosis: Annotated[Literal["Yes", "No"], Form(description="Has the patient been previously diagnosed with scoliosis or any other spinal condition?")] = "No",
    family_history: Annotated[Literal["Yes", "No"], Form(description="Is there a family history of scoliosis or spinal issues for the patient?")] = "No",
    past_injuries: Annotated[Literal["Yes", "No"], Form(description="Has the patient had any back or spinal injuries in the past?")] = "No",
    physical_activity_level: Annotated[Literal["Sedentary", "Lightly active", "Moderately active", "Very active"], Form(description="How would you describe the patient's regular physical activity level?")] = "Moderately active",
    screen_time: Annotated[str, Form(description="How many hours per day does the patient spend sitting or using screens?")] = "6",
    posture_awareness: Annotated[Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], Form(description="How would you rate the patient's awareness of maintaining good posture? (1 to 10)")] = "5",
    ergonomic_setup: Annotated[Literal["Yes", "No"], Form(description="Does the patient have an ergonomic workspace setup (e.g., chair, desk height)?")] = "Yes",
    sleeping_position: Annotated[Literal["Back", "Side", "Stomach", "Other"], Form(description="What is the patient's usual sleeping position?")] = "Side",
    shoulder_alignment: Annotated[Literal["Yes", "No"], Form(description="Does the patient have one shoulder higher than the other?")] = "No",
    head_alignment: Annotated[Literal["Yes", "No"], Form(description="Does the patient's head appear to lean to one side when viewed from the front?")] = "No",
    spinal_curvature: Annotated[Literal["Yes", "No"], Form(description="Does the patient feel or notice any unevenness in their spine?")] = "No",
    hip_level: Annotated[Literal["Yes", "No"], Form(description="Are the patient's hips level when standing straight?")] = "Yes",
    foot_alignment: Annotated[Literal["Straight", "Inward", "Outward"], Form(description="Do the patient's feet point straight ahead or turn inward/outward?")] = "Straight",
    clothes_fit: Annotated[Literal["Yes", "No"], Form(description="Do the patient's clothes fit unevenly, indicating potential body asymmetry?")] = "No",
    mobility: Annotated[Literal["Yes", "No"], Form(description="Does the patient experience difficulty in moving or performing daily tasks due to posture?")] = "No",
    fatigue: Annotated[Literal["Yes", "No"], Form(description="Does the patient often feel fatigued or tired, even without strenuous activity?")] = "No",
    breathing: Annotated[Literal["Yes", "No"], Form(description="Has the patient's posture affected their breathing patterns?")] = "No",
    balance_issues: Annotated[Literal["Yes", "No"], Form(description="Does the patient experience balance problems?")] = "No",
    image: UploadFile = File(...),
):
    score = 0
    hip_angle = None
    image_id = f"{uuid.uuid4()}.png"
    image_path = os.path.join("processed_images", image_id)

    os.makedirs("processed_images", exist_ok=True)

    img_bytes = image.file.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_np = np.array(pil_image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    annotated_image = image_np.copy()

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            def get_coords(lm):
                return [lm.x, lm.y]

            shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
            knee = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_KNEE])

            def calc_angle(a, b, c):
                a, b, c = map(np.array, (a, b, c))
                radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians * 180.0 / np.pi)
                return 360 - angle if angle > 180 else angle

            hip_angle = calc_angle(shoulder, hip, knee)

            if hip_angle < 165:
                score += 2

    if pain_present == "Yes":
        score += 2
        if int(pain_severity) >= 7:
            score += 2
        elif int(pain_severity) >= 4:
            score += 1
        if pain_duration in ["6 months to a year", "Over a year"]:
            score += 2
        if symptom_onset == "Suddenly":
            score += 1
        if activity_related_pain == "Yes":
            score += 1

    if previous_diagnosis == "Yes":
        score += 3
    if family_history == "Yes":
        score += 2
    if past_injuries == "Yes":
        score += 1
    if physical_activity_level in ["Sedentary", "Lightly active"]:
        score += 1
    if int(screen_time) > 6:
        score += 1
    if int(posture_awareness) < 5:
        score += 1
    if ergonomic_setup == "No":
        score += 1
    if sleeping_position == "Stomach":
        score += 1
    if shoulder_alignment == "Yes":
        score += 2
    if head_alignment == "Yes":
        score += 2
    if spinal_curvature == "Yes":
        score += 3
    if hip_level == "No":
        score += 2
    if foot_alignment != "Straight":
        score += 1
    if clothes_fit == "Yes":
        score += 1
    if mobility == "Yes":
        score += 2
    if fatigue == "Yes":
        score += 1
    if breathing == "Yes":
        score += 1
    if balance_issues == "Yes":
        score += 1

    risk = "Low"
    if score >= 15:
        risk = "High"
    elif score >= 8:
        risk = "Moderate"

    # Save annotated image
    Image.fromarray(annotated_image).save(image_path)

    base_url = str(request.base_url)
    return AssessmentOutput(
        risk_level=risk,
        score=score,
        hip_angle=hip_angle,
        processed_image_link=f"{base_url}processed_images/{image_id}"
    )
