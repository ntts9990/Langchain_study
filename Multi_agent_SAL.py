import os
import functools
import random
from collections import OrderedDict
from typing import Callable, List

import tenacity
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.prompts import (
    PromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain.globals import set_debug

set_debug(True)

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = "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"

openai_api_key = os.getenv("OPENAI_API_KEY")


def print_w(text):
    # 파일에 텍스트 작성
    with open("target_sal_eng_test1.txt", 'a') as file:  # 'a' 모드를 사용하여 파일 끝에 추가
        file.write(text + "\n")
    file.close()

class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # 시간 증가
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 다음 발언자 선택
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 다음 발언자가 메시지를 보냄
        message = speaker.send()

        # 3. 모든 에이전트가 메시지를 수신
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 시간 증가
        self._step += 1

        return speaker.name, message

class IntegerOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."


class DirectorDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        model: ChatOpenAI,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.next_speaker = ""

        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = "Finish the conversation by stating a concluding message and thanking everyone."
        self.continuation_clause = "Do not end the conversation. Keep the conversation going by adding your own ideas."

        # 1. 이전 발언자에게 응답을 생성하기 위한 프롬프트
        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}

Follow up with an insightful comment.
{{termination_clause}}
{self.prefix}
        """,
        )

        # 2. 다음에 말할 사람을 결정하기 위한 프롬프트
        self.choice_parser = IntegerOutputParser(
            regex=r"<(\d+)>", output_keys=["choice"], default_output_key="choice"
        )
        self.choose_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "speaker_names"],
            template=f"""{{message_history}}

Given the above conversation, select the next speaker by choosing index next to their name: 
{{speaker_names}}

{self.choice_parser.get_format_instructions()}

Do nothing else.
        """,
        )

        # 3. 다음 발언자에게 말하도록 유도하는 프롬프트
        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}

The next speaker is {{next_speaker}}. 
Prompt the next speaker to speak with an insightful question.
{self.prefix}
        """,
        )

    def _generate_response(self):
        # 만약 self.stop = True이면, termination clause으로 프롬프트를 주입합니다
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability

        print_w(f"\tStop? {self.stop}\n")

        response_prompt = self.response_prompt_template.format(
            message_history="\n".join(self.message_history),
            termination_clause=self.termination_clause if self.stop else "",
        )

        self.response = self.model(
            [
                self.system_message,
                HumanMessage(content=response_prompt),
            ]
        ).content

        return self.response

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_none(),  # 재시도 간 대기 시간 없음
        retry=tenacity.retry_if_exception_type(ValueError),
        before_sleep=lambda retry_state: print_w(
            f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
        ),
        retry_error_callback=lambda retry_state: 0,
    )  # 모든 재시도가 소진될 때 기본 값
    def _choose_next_speaker(self) -> str:
        speaker_names = "\n".join(
            [f"{idx}: {name}" for idx, name in enumerate(self.speakers)]
        )
        choice_prompt = self.choose_next_speaker_prompt_template.format(
            message_history="\n".join(
                self.message_history + [self.prefix] + [self.response]
            ),
            speaker_names=speaker_names,
        )

        choice_string = self.model(
            [
                self.system_message,
                HumanMessage(content=choice_prompt),
            ]
        ).content
        choice = int(self.choice_parser.parse(choice_string)["choice"])

        return choice

    def select_next_speaker(self):
        return self.chosen_speaker_id

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        # 1. 이전 발언자에게 응답을 생성하고 저장
        self.response = self._generate_response()

        if self.stop:
            message = self.response
        else:
            # 2. 다음에 말할 사람을 결정
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print_w(f"\tNext speaker: {self.next_speaker}\n")

            # 3. 다음 발언자에게 말하도록 유도
            next_prompt = self.prompt_next_speaker_prompt_template.format(
                message_history="\n".join(
                    self.message_history + [self.prefix] + [self.response]
                ),
                next_speaker=self.next_speaker,
            )
            message = self.model(
                [
                    self.system_message,
                    HumanMessage(content=next_prompt),
                ]
            ).content
            message = " ".join([self.response, message])

        return message
    

topic = "Your goal is to quickly identify customer types and use sales tactics appropriately to 'engage with customers via online chat to understand their specific needs and steer the conversation to quickly lead to a purchase.' Throughout the process, you should actively leverage your areas of expertise and communicate with each other. The whole process should be documented."
director_name = "Steve"
agent_summaries = OrderedDict(
    {
        "DH": ("Project Director", "Project Manager"),
        "SJ": ("Create 5 consecutive predicted statements that a customer would say to a chatbot on an online shopping mall.", "Customer statements"),
        "SL": ("Check the customer's statements and summarize them to keep only the important parts.", "Intent summarizer"),
        "MS": ("Reads the customer's statements, summarized intent, and determines the customer type: 'hesitant', 'denial', 'doubtful', or 'seeking confirmation'.", "Customer type categorizer"),
        "SH": ("Checks the customer's outburst, summarized intent, and customer type and comes up with an appropriate sales strategy, focusing on 'light conversation and ongoing engagement' and 'flexibility and adaptability'.", "Sales strategy presenter"),
    }
)

agent_summary_string = "\n- ".join(
    [""]
    + [
        f"{name}: {role}, located in {location}"
        for name, (role, location) in agent_summaries.items()
    ]
)

conversation_description = f"""This is a conversation to accomplish this topic: {topic}.
This discussion features a team of experts, including {agent_summary_string}, who offer unique insights and expertise in online non-face-to-face sales marketing. You will be asked to provide a five-sentence opening statement, an intent summary, a customer type judgment and conviction level (%), and a sales strategy based on the preceding factors."""
agent_descriptor_system_message = SystemMessage(
    content="Talk about the specifics of the design based on each team member's role."
)


def generate_agent_description(agent_name, agent_role, agent_location):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please provide a professional description of {agent_name}, the person in charge of {agent_location} who does {agent_role}, highlighting their duties and responsibilities.
            Speak directly to {agent_name}
            Do not add anything else."""
        ),
    ]
    agent_description = ChatOpenAI(model="gpt-4", temperature=1.0)(agent_specifier_prompt).content
    return agent_description


def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f"""{conversation_description}
Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.
Your description is as follows: {agent_description}
You are discussing the topic: {topic}.
Your goal is to provide the most informative, professional, and effective conversation strategy for the topic from the perspective of your job and the person you're talking to.
"""


def generate_agent_system_message(agent_name, agent_header):
    return SystemMessage(
        content=(
            f"""{agent_header}
You will speak in the style of {agent_name}, and exaggerate your personality.
You should have specific programming design patterns, file and folder structures, and names and functional descriptions of classes and functions.
Do not say the same things over and over again.
Speak in the first person from the perspective of {agent_name}
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {agent_name}.
Stop speaking the moment you finish speaking from your perspective.
Do not add anything else.
    """
        )
    )


agent_descriptions = [
    generate_agent_description(name, role, location)
    for name, (role, location) in agent_summaries.items()
]
agent_headers = [
    generate_agent_header(name, role, location, description)
    for (name, (role, location)), description in zip(
        agent_summaries.items(), agent_descriptions
    )
]
agent_system_messages = [
    generate_agent_system_message(name, header)
    for name, header in zip(agent_summaries, agent_headers)
]

for name, description, header, system_message in zip(
    agent_summaries, agent_descriptions, agent_headers, agent_system_messages
):
    print_w(f"\n\n{name} Description:")
    print_w(f"\n{description}")
    print_w(f"\nHeader:\n{header}")
    print_w(f"\nSystem Message:\n{system_message.content}")

topic_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(
        content=f"""{conversation_description}
        Tell us more about your topic. 
        Organize your topic into one question that needs to be answered.
        Demonstrate your subject matter expertise.
        Stick to the specified topic.
        Don't add anything else."""
    ),
]
specified_topic = ChatOpenAI(model="gpt-4", temperature=1.0)(topic_specifier_prompt).content

print_w(f"Original topic:\n{topic}\n")
print_w(f"Detailed topic:\n{specified_topic}\n")

def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    """
    If the step is even, then select the director
    Otherwise, the director selects the next speaker.
    """
    # 디렉터는 홀수 스텝에서만 말합니다.
    if step % 2 == 1:
        idx = 0
    else:
        # 여기에서 디렉터는 다음 발언자를 선택합니다.
        idx = director.select_next_speaker() + 1  # +1은 디렉터를 제외하기 때문입니다.
    return idx

director = DirectorDialogueAgent(
    name=director_name,
    system_message=agent_system_messages[0],
    model=ChatOpenAI(model="gpt-4", temperature=0.2),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.05,
)

agents = [director]
for name, system_message in zip(
    list(agent_summaries.keys())[1:], agent_system_messages[1:]
):
    agents.append(
        DialogueAgent(
            name=name,
            system_message=system_message,
            model=ChatOpenAI(model="gpt-4", temperature=0.2),
        )
    )

    simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject("Audience member", specified_topic)
print_w(f"(Audience member): {specified_topic}")
print_w("\n")

while True:
    name, message = simulator.step()
    print_w(f"({name}): {message}")
    print_w("\n")
    if director.stop:
        break