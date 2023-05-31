from .prompt_common import Participant, Message, ParticipantNames
from typing import List

participant_names: ParticipantNames = {
  i.name: i.value for i in Participant
}

def mpt_prompt(
  history: List[Message],
  participant_names: ParticipantNames = participant_names,
) -> str:
  history_str: str = ''.join([
    f"<|im_start|>{participant_names[participant]}\n{message}<|im_end|>"
    for participant, message in history
  ])
  chat_to_complete = f"{history_str}<|im_start|>{participant_names[Participant.Assistant]}\n"
  return chat_to_complete