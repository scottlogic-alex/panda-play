from typing import NamedTuple, TypedDict
from enum import Enum

class ParticipantNames(TypedDict):
  user: str
  assistant: str

class Participant(Enum):
  User = 'user'
  Assistant = 'assistant'
  System = 'system'

class Message(NamedTuple):
  participant: Participant
  message: str