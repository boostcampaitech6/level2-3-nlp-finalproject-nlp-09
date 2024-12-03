# CHAT PROCESS의 흐름에 따라 분기에 맞는 질문들을 구현
# 각 분기마다 나올 수 있는 질문은 한정되어 있음
# `random` 모듈을 활용해서 random 선택 구현
import random
import time

# 1분기 특별한 하루가 없었을 경우 -> 감정 질문
FIRST_CHATS_NOT_SPECIAL = [
    "그렇구나. 그럼 오늘 하루의 감정은 어땠어?",
    "딱히 특별한 일은 없었구나! 그럼 오늘 하루 느꼈던 감정에 대해 말해줄 수 있어?",
    "그런 날이 있을 수도 있지! 그렇다면 감정에 대한 이야기를 해볼까? 오늘 어떤 감정이었던 것 같아?",
    "그럼 오늘 어떤 감정으로 하루를 보냈어?",
    "하루하루가 다 특별할 수는 없지! 그러면 오늘 하루는 어떤 감정을 느꼈던 하루였어?"
]

# 2분기 감정이 없었을 경우 -> RANDOM 질문
SECOND_CHATS_NOT_SPECIAL = [
    "쉬는 날에 주로 어떤 일을 하며 하루를 보내?",
    "가족과 함께 저녁을 먹으러 간다면 어디가 좋을까?",
    "친구와 함께 놀러간다면 어디로 놀러가고 싶어?",
    "야식을 먹는다면 무슨 메뉴를 먹어야 할까?",
    "여행을 간다면 어디가 가장 좋을 것 같아?"
]

def get_random_first_chat():
    random.seed(int(time.time()))
    return random.choice(FIRST_CHATS_NOT_SPECIAL)

def get_random_second_chat():
    random.seed(int(time.time()))
    return random.choice(SECOND_CHATS_NOT_SPECIAL)