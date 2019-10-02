# smart_alarm_clock_project
with OpenCV


## 1. 개요 

사용자가 잠에서 깨어, 카메라에 얼굴을 인식시켜야만 알람이 꺼지는 알람시계

## 2. 필요성 및 동기

알람이 울리면 바로 일어나지 않고, 10분씩 알람시간을 미루곤 했는데 종종 알람 소리를 듣지 못한 채 늦잠을 자버릴 때가 있었다.
재미도 있으면서, 늦잠을 방지하고, 규칙적인 생활에 도움이 되기위해 구현해 보았다.

## 3. 구현 환경

1) window10 Home
2) Python 3.6
3) OpenCV-python library 4.1.0 
4) Numpy 1.16
5) pygame 1.9.6

## 4. 동작 과정

1) 알람 시간 설정
2) 알람 시간이 되면 알람 시작 및 모션 감지 시작
3) 움직임이 감지되면 얼굴 인식 시작
4) 얼굴이 인식되면 알람 종료

## 5. 개선해야할 부분

1) 얼굴인식 후 다시 잠드는 것을 막기위해 처리 필요

2) 손동작을 통한 알람시간 연장 기능 필요

3) 편리함을 위해 gui 버젼이 필요

 
