import cv2 as cv
import numpy as np
import time
from pygame import mixer 

'''
openCV를 이용한 알람시계
현재 프로그램은 tui version

기능 :
1) 사용자가 일어날 때 까지 ALARM! // 움직임 감지 -> 얼굴인식 시작 -> 얼굴 인식 O -> 알람 정지 -> 눕는지 확인
2) 손동작을 인식해서 시간 추가! ( 5분 , 10분 등)
3) 사용자가 일어났지만 일정 시간 내에 다시 누웠을 경우, 다시 ALARM!


@update
tinker를 사용해서 gui로 업데이트할 예정
gui ver.에서 필요한 H/W :
1) rasberry pi 
2) bluetooth speaker
3) pi camera
4) small size lcd (5" or 7")

'''


# 알람 시간 설정
def set_data(hour = 0, minute = 0):
    print("|||-----ALARM CLOCK-----|||")
    
    # 날짜 format code 
    # %c 
    # Sat      May     19   11:14:27 2018
    c_week, c_month, c_day, c_time, _ = time.strftime('%c', time.localtime(time.time())).split()
    

    print(f"현재 시간 >> {c_time}")
            
    try:
        hour, minute = map(int,input(
        '''
알람 시간을 설정합니다.
ex)  07:30 AM에 알람 설정하려면, 07:30 입력 >> ''').split(':')) 
    
    except:
        print("|--------------------------시간은 hh:mm 형식으로 입력---------------------------|")
        set_data()
        
    
    
    # c = current
    c_hour, c_min, _ = map(int, c_time.split(':'))

    # a = alarm
    a_hour = (24 - c_hour) + hour
    a_min = (60 - c_min) + minute
    
    # 시간 계산
    carry_hour, a_hour  = divmod(a_hour, 24)
    carry_min, a_min = divmod(a_min, 60)    
    
    print("%d시 %d분 후에 알람이 울립니다."% (a_hour, a_min)) 
    

    
    return hour, minute




# 알람 기능
def alarm(repeat = 0):
    print("***ALARM START***")
    mixer.init()
    mixer.music.load('./Cover_Girl.mp3')
    mixer.music.play(repeat)


# opencv 를 이용한 움직임 검출
def face_detection():
    print('START FACE DETECTION')
    cap = cv.VideoCapture(0)
    while(1):
        ret, img = cap.read()

        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        wake_up = 0
        
        for (x,y,w,h) in faces:
            img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            wake_up += 1
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        cv.imshow('face_detection', img)        
        k = cv.waitKey(30)
        if k == ord('s'):
            break
        
        if wake_up >= 1:
            break
        
    cap.release()
    cv.destroyAllWindows()


# opencv mog2()를 이용한 움직임 검출
def hand_gesture():
    pass



# opencv mog2()를 이용한 움직임 검출
def motion_detection():
    print("START MOTION DETECTION && ALARM")
    
    
    
    '''
    history : 히스토리 길이. (기본값 = 500)
    varThreshold :이 값은 픽셀이 배경 모델을 잘 묘사하는지 여부를 결정. (기본값 = 16)
    detectShadows : frame에서 그림자가 중요한지 여부.
    '''
    cap = cv.VideoCapture(0)
    fgbg = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=200, detectShadows=0)

    motion_count = 0


    
    while(1):
        # 알람 시작, 움직임이 감지되면 얼굴인식 시작, 얼굴 인식 되면 알람 정지

        ret, frame = cap.read()

        width = frame.shape[1]
        height = frame.shape[0]
        
        frame = cv.resize(frame, (640, 480))
        fgmask = fgbg.apply(frame)

        """
        cv.connectedComponentsWithStats(image)
        입력 인자 image : 레이블링할 이미지        
        
        출력
        nlabels : 레이블의 개수
        labels : 레이블링한 결과 이미지
        stats : 레이블링 된 이미지 배열
        centroids : 레이블링 된 이미지의 중심좌표
        """
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(fgmask)

        for index, centroid in enumerate(centroids):
            # 레이블링 된 이미지의 x좌표가 같거나 중심점이 없을 때(안잡혔을 때)는 넘어간다
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue


            x, y, width, height, area = stats[index]
            centerX, centerY = int(centroid[0]), int(centroid[1])

            # 움직인 면적이 적당히 커야 화면에 표시
            if area > 200:
                cv.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
                motion_count += 1
        
        cv.imshow('mask',fgmask)
        cv.imshow('frame',frame)
        
        if motion_count >= 10:
            break
        
        # s 키를 눌러서 루프 종료
        k = cv.waitKey(30)
        if k == ord('s'):
            break
        
        # 0.5초마다 움직임 체크 // 연산량을 줄이기 위해서
        time.sleep(0.5)
    cap.release()
    cv.destroyAllWindows()
    
    # 움직임이 감지되면 얼굴 감지로 이동
    face_detection()
    

def main():
    # 알람 시간 세팅
    alarm_hour, alarm_min = set_data()
    while 1:
        # 알람 시간이 될 때 까지 대기
        current_hour, current_min, _ = map(int, time.strftime('%X', time.localtime(time.time())).split(':'))    
        print("현재 시간 >>", time.strftime('%X', time.localtime(time.time())) + "\t\t", end = "")
        print("알람 시간 >>", alarm_hour, alarm_min, "00" ,sep = ':')
        # 알람 시간이 되면, 알람 시작 && 모션 감지 시작
        if (current_hour == alarm_hour) and (current_min == alarm_min) :
            alarm(-1)
            motion_detection() 
            break
        time.sleep(1)
main()