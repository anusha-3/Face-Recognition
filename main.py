import cv2
import numpy as np
import face_recognition as fr
import os
import streamlit as st

st.title("Face Recognition System")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
col1, col2 = st.columns([1, 1])
with col1:
    button = st.button('Play')
with col2:
    STOP = st.button('Stop')
while(button):
    BOX = st.image([])
    path = 'images'
    images = []
    personName = []
    myList = os.listdir(path)
# print(myList)

    for cu_img in myList:
        current_img = cv2.imread(f'{path}/{cu_img}')
        images.append(current_img)
        personName.append(os.path.splitext(cu_img)[0])
# print(personName)

    def faceEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = fr.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = faceEncodings(images)
    camera = cv2.VideoCapture(0)
# print("All Encodings Completed!!!")

# camera = cv2.VideoCapture(1)

    while (True):
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

        facesCurrentFrame = fr.face_locations(faces)
        encodeCurrentFrame = fr.face_encodings(
            faces, facesCurrentFrame)

        for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
            matches = fr.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = fr.face_distance(
                encodeListKnown, encodeFace)
            name = "Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2),
                          (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = personName[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2-35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        BOX.image(frame)
        cv2.imshow('frame', frame)
        if STOP:
            st.stop
    # FRAME_WINDOW.image(frame)

# else:
    # st.write('Stopped')
