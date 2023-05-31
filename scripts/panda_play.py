from dataclasses import dataclass, field, fields, make_dataclass
from typing import Optional, TypedDict, NamedTuple, List, Dict, Callable, Type, Tuple
import torch
from torch import LongTensor
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  HfArgumentParser,
  set_seed,
  StoppingCriteria,
  StoppingCriteriaList,
  PreTrainedTokenizerBase,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast
)
import logging
from enum import Enum
from src.callback_text_iterator_streamer import CallbackTextIteratorStreamer
from src.falcon_prompt import falcon_prompt
from src.mpt_prompt import mpt_prompt
from src.prompt_common import Message, Participant

logger = logging.getLogger(__name__)

class TokenizerOutput(TypedDict):
  input_ids: LongTensor
  attention_mask: LongTensor

class ChatStyle(Enum):
  MPT = 'MPT'
  Falcon = 'Falcon'

class SufficientResponse(BaseException): ...

@dataclass
class StopOnTokens(StoppingCriteria):
  stop_token_ids: List[int]
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in self.stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
    default="tiiuae/falcon-7b-instruct"
  )
  trust_remote_code: Optional[bool] = field(
    default=False,
    metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
  )
  double_quant: bool = field(
    default=True,
    metadata={"help": "Compress the quantization statistics through double quantization."}
  )
  quant_type: str = field(
    default="nf4",
    metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
  )
  bits: int = field(
    default=4,
    metadata={"help": "How many bits to use."}
  )
  bf16: Optional[bool] = field(
    default=False,
    metadata={"help": "Compute type of the model. If quantizing: this is also the compute type used for quantized computations. Prefer to turn this on if you are quantizing and your GPU supports it. You probably also want it even if you're not quantizing."}
  )
  context_length: int = field(
    default=2048,
    metadata={"help": "How many tokens to include in context."}
  )

@dataclass
class TokenizerArguments:
  use_fast: bool = field(
    default=True,
    metadata={"help": "Recommend True for MPT, False for Falcon and PandaLM"}
  )

@dataclass
class ChatArguments:
  chat_style: str = field(
    default=ChatStyle.MPT,
    metadata={"help": "Prompt template with which to lay out the conversation thread"}
  )
  your_name: str = field(
    default='user',
    metadata={"help": "Your name in the chat log."}
  )
  bot_name: str = field(
    default='assistant',
    metadata={"help": "Chatbot's name in the chat log."}
  )
  system_prompt: str = field(
    default='',
    metadata={"help": "Influence how the chatbot responds, by seeding the conversation with some context."}
  )

@dataclass
class SamplingArguments:
  overrun_countermeasures: bool = field(
    default=False,
    metadata={"help": "[Recommended for Falcon, but not for MPT] Detect when bot is about to start talking to itself; end the generation before that happens. The bot is *supposed* to emit an end-of-sentence token to indicate that it's finished its reply, but very often it neglects to do this, and continues to sequence-complete the conversation. Hence this countermeasure tries to detect and prevent that."}
  )
  trim_leading_whitespace: bool = field(
    default=False,
    metadata={"help": "[Recommended for Falcon, but not for MPT] Don't allow bot to start a reply with a space or a line break."}
  )

@dataclass
class MiscArguments:
  seed: Optional[int] = field(
    default=64,
    metadata={"help": "Random seed, for deterministic generation."}
  )
  compile: bool = field(
    default=False,
    metadata={"help": "Invoke torch.compile() on the model, with mode='max-autotune'. Requires PyTorch 2, CUDA, and either Python 3.10 or Python 3.11 with a recent torch nightly. Will make the first inference from the model take a bit longer, but subsequent inferences will be faster."}
  )

@dataclass
class GenerationArguments:
  # For more hyperparameters check:
  # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
  # Length arguments
  max_new_tokens: Optional[int] = field(
    default=256,
    metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                      "if predict_with_generate is set."}
  )
  min_new_tokens: Optional[int] = field(
    default=None,
    metadata={"help": "Minimum number of new tokens to generate."}
  )

  # Generation strategy
  # do_sample: Optional[bool] = field(default=True)
  num_beams: Optional[int] = field(default=1)
  num_beam_groups: Optional[int] = field(default=1)
  penalty_alpha: Optional[float] = field(default=None)
  use_cache: Optional[bool] = field(default=True)

  # Hyperparameters for logit manipulation
  temperature: Optional[float] = field(default=1.0)
  top_k: Optional[int] = field(default=10)
  top_p: Optional[float] = field(default=1.0)
  typical_p: Optional[float] = field(default=1.0)
  diversity_penalty: Optional[float] = field(default=0.0)
  repetition_penalty: Optional[float] = field(default=1.0)
  length_penalty: Optional[float] = field(default=1.0)
  no_repeat_ngram_size: Optional[int] = field(default=0)

def get_model(args: ModelArguments) -> AutoModelForCausalLM:
  config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=args.trust_remote_code,
  )
  config.update({"max_seq_len": args.context_length}) # was originally trained on 2048
  cuda_avail = torch.cuda.is_available()
  compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
  load_in_4bit = args.bits == 4 and cuda_avail
  load_in_8bit = args.bits == 8 and cuda_avail

  quantization_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.double_quant,
    bnb_4bit_quant_type=args.quant_type,
  ) if cuda_avail else None

  if not cuda_avail:
    logger.warning("You don't have CUDA, so we have turned off quantization. If you happen to be on a Mac: you probably have enough unified memory to run in fp16 anyway…")

  if compute_dtype == torch.float16 and cuda_avail and torch.cuda.is_bf16_supported():
    print("Your GPU supports bfloat16; you may want to try it with --bf16 (note: I'm not sure how important this is for inference, but it's certainly preferred when training with 4-bit quantization.)")

  device_map = {'': 'mps'} if torch.backends.mps.is_available() else 'auto'
  model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    quantization_config=quantization_config,
    torch_dtype=compute_dtype,
    trust_remote_code=args.trust_remote_code,
  ).eval()
  model.config.torch_dtype=compute_dtype

  return model

def main():
  derived_dclasses = Tuple[List[Type[ModelArguments]], Type[GenerationArguments], Type[TokenizerArguments], Type[ChatArguments]] = tuple([make_dataclass(
    f'{instance}{dclass.__name__}',
    fields=[(f'm0_{field.name}', field.type, field) for field in fields(dclass)]
  ) for instance in ('M0', 'M1', 'Judge')] for dclass in (ModelArguments, GenerationArguments, TokenizerArguments, ChatArguments))

  model_arg_dclasses, gen_arg_dclasses, tokenizer_arg_dclasses, chat_arg_dclasses = derived_dclasses
  
  # M0ModelArguments = make_dataclass('M0ModelArguments', fields=[(f'm0_{field.name}', field.type, field) for field in fields(ModelArguments)])
  # M1ModelArguments = make_dataclass('M1ModelArguments', fields=[(f'm1_{field.name}', field.type, field) for field in fields(ModelArguments)])
  # JudgeModelArguments = make_dataclass('JudgeModelArguments', fields=[(f'j_{field.name}', field.type, field) for field in fields(ModelArguments)])

  # M0GenerationArguments = make_dataclass('M0GenerationArguments', fields=[(f'm0_{field.name}', field.type, field) for field in fields(GenerationArguments)])
  # M1GenerationArguments = make_dataclass('M1GenerationArguments', fields=[(f'm1_{field.name}', field.type, field) for field in fields(GenerationArguments)])
  # JudgeGenerationArguments = make_dataclass('JudgeGenerationArguments', fields=[(f'j_{field.name}', field.type, field) for field in fields(GenerationArguments)])

  hfparser = HfArgumentParser((
    # M0ModelArguments, M1ModelArguments, JudgeModelArguments,
    # M0GenerationArguments, M1GenerationArguments, JudgeGenerationArguments,
    *model_arg_dclasses,
    *gen_arg_dclasses,
    *tokenizer_arg_dclasses,
    *chat_arg_dclasses,
    MiscArguments,
  ))

  (m0_model_args, m1_model_args, j_model_args,
   m0_gen_args, m1_gen_args, j_gen_args,
   m0_tok_args, m1_tok_args, j_tok_args,
   m0_chat_args, m1_chat_args, j_chat_args,
   misc_args, extra_args) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

  if extra_args:
    raise f"Received unsupported command-line args: {extra_args}"

  gen_configs: List[GenerationConfig] = [GenerationConfig(**vars(gen_args)) for gen_args in (m0_gen_args, m1_gen_args, j_gen_args)]
  m0_gen_config, m1_gen_config, j_gen_config = gen_configs

  # tokenizer = AutoTokenizer.from_pretrained("WeOpenML/PandaLM-7B-v1",use_fast=False)

  # model = AutoModelForCausalLM.from_pretrained("WeOpenML/PandaLM-7B-v1")
  # judge: AutoModelForCausalLM = get_model()

  models: List[AutoModelForCausalLM] = [get_model(model_args).cpu() for model_args in (m0_model_args, m1_model_args, j_model_args)]
  model0, model1, judge = models

  set_seed(misc_args.seed)
  if misc_args.compile:
    for model in (model0, model1, judge):
      torch.compile(model, mode='max-autotune')
  
  tokenizers: List[PreTrainedTokenizer|PreTrainedTokenizerFast] = [AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    padding=False,
    use_fast=tok_args.use_fast,
    # add_special_tokens=False,
  ) for model_args, tok_args in zip((m0_model_args, m1_model_args, j_model_args), (m0_tok_args, m1_tok_args, j_tok_args))]
  m0_tok, m1_tok, j_tok = tokenizers

  for gen_config, tokenizer in zip(gen_configs, tokenizers):
    gen_config.eos_token_id = gen_config.pad_token_id = tokenizer.eos_token_id

  stop = StopOnTokens([tokenizer.eos_token_id])
  stopping_criteria=StoppingCriteriaList([stop])

  history: List[Message] = [Message(Participant.System, misc_args.system_prompt)] if misc_args.system_prompt else []

  participant_names: Dict[Participant, str] = {
    Participant.User: misc_args.your_name,
    Participant.Assistant: misc_args.bot_name,
  }

  first = False
  history += [Message(Participant.User, user_input)]

  chat_to_complete: str = '\n'.join([
    *[
      f"{'' if participant is Participant.System else f'{participant_names[participant]}: '}{message}"
      for participant, message in history
    ],
    f'{participant_names[Participant.Assistant]}:'
  ])

  tokenized_prompts: TokenizerOutput = tokenizer(chat_to_complete, return_tensors='pt', padding=False, add_special_tokens=False)

  response = ''
  if misc_args.trim_leading_whitespace:
    strip_whitespace: Callable[[str], str] = lambda x: x.lstrip() if response == '' else x
  else:
    strip_whitespace: Callable[[str], str] = lambda x: x

  if misc_args.overrun_countermeasures:
    # the model may continue adding to the conversation (replying to itself) instead of emitting an EOS token.
    # we try to intercept this. If it looks like it's starting a new message in the voice of either of the chat participants: don't print that, and stop generation.
    acc_overrun = ''

    def on_text(message: str, stream_end = False):
      nonlocal response, acc_overrun

      overrun_and_message = f'{acc_overrun}{message}'

      newline_ix = overrun_and_message.find('\n')
      if newline_ix > -1:
        pre_newline, post_newline = overrun_and_message.split('\n', maxsplit=1)

        potential_participant_name = post_newline[:overrun_and_message.find(':')]
        if potential_participant_name == f'{misc_args.your_name}:' or potential_participant_name == f'{misc_args.bot_name}:':
          raise SufficientResponse()
        if potential_participant_name.rstrip(f'{misc_args.your_name}:') == '' or potential_participant_name.rstrip(f'{misc_args.bot_name}:') == '':
          # could potentially grow to match one of the names. we need to accumulate to see whether that's where the bot was going.
          acc_overrun = f'\n{post_newline}'

          addendum: str = strip_whitespace(pre_newline)
          response += addendum
          print(addendum, end='', flush=True)
          return
        # the potential_participant_name cannot grow into a reply from either chat participant, so this must be something else. flush everything we accumulated.

      addendum: str = strip_whitespace(overrun_and_message)
      response += addendum
      print(addendum, end='', flush=True)
      acc_overrun = ''
  else:
    def on_text(message: str, stream_end = False):
      nonlocal response
      addendum: str = strip_whitespace(message)
      response += addendum
      print(addendum, end='', flush=True)

  try:
    prediction: LongTensor = model.generate(
      input_ids=tokenized_prompts.input_ids.to(model.device),
      attention_mask=tokenized_prompts.attention_mask.to(model.device),
      generation_config=generation_config,
      do_sample=generation_config.temperature > 0.,
      stopping_criteria=stopping_criteria,
    )
    # if you wanted to see the result, you can do so like this:
    #   decode: List[str] = tokenizer.decode(prediction[0,tokenized_prompts.input_ids.size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # but we're already streaming it to the console via our callback
  except (KeyboardInterrupt, SufficientResponse):
    pass

if __name__ == "__main__":
  main()