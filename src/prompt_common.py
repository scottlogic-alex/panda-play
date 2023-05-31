from typing import Any, NamedTuple, TypedDict, Protocol, List
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

class HistoryToPrompt:
  def __call__(self, 
    history: List[Message],
    participant_names: ParticipantNames,
  ) -> str: ...