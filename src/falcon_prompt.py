from .prompt_common import Participant, Message, ParticipantNames
from typing import List

participant_names: ParticipantNames = {
  i.name: i.value for i in Participant
}

def falcon_prompt(
  history: List[Message],
  participant_names: ParticipantNames = participant_names,
) -> str:
  chat_to_complete: str = '\n'.join([
    *[
      f"{'' if participant is Participant.System else f'{participant_names[participant]}: '}{message}"
      for participant, message in history
    ],
    f'{participant_names[Participant.Assistant]}:'
  ])
  return chat_to_complete