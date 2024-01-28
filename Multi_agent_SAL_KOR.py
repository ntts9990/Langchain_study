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
    with open("target_sal_test_3.5.txt", 'a') as file:  # 'a' 모드를 사용하여 파일 끝에 추가
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
        self.message_history = ["지금까지의 대화 내용입니다."]

    def send(self) -> str:
        """
        메시지 기록에 대한 챗모델을 적용하고 메시지 문자열을 반환합니다.
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
        메시지 기록에 {name}의 {message}를 추가합니다.
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
        {name}으로부터 시작하는 {message}로 대화를 시작합니다.
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
        return "응답은 이와 같은 꺽쇠 괄호 안에 정수로 표현되어야 합니다: <int>."


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
        self.termination_clause = "종결 메시지를 말하고 모든 사람에게 감사함을 표시함으로써 대화를 마무리합니다."
        self.continuation_clause = "대화를 종료하지 마세요. 자신의 아이디어를 추가하여 대화를 이어가세요."

        # 1. 이전 발언자에게 응답을 생성하기 위한 프롬프트
        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}

통찰력 있는 논평(comment)으로 이어가세요..
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

위 대화를 바탕으로 다음 발언자를 선택하세요. 이름 옆의 인덱스를 선택하세요: 
{{speaker_names}}

{self.choice_parser.get_format_instructions()}

다른 것은 하지 마세요.
        """,
        )

        # 3. 다음 발언자에게 말하도록 유도하는 프롬프트
        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}

다음 발언자는 {{next_speaker}}입니다.
통찰력 있는 질문으로 다음 발언자에게 말하도록 유도하세요.
{self.prefix}
        """,
        )

    def _generate_response(self):
        # 만약 self.stop = True이면, termination clause으로 프롬프트를 주입합니다
        sample = random.uniform(0, 1)
        self.stop = sample < self.stopping_probability

        print_w(f"\t중단하나요? {self.stop}\n")

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
            f"ValueError 발생: {retry_state.outcome.exception()}, 재시도 중..."
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
        메시지 기록에 챗모델을 적용하여 메시지 문자열을 반환합니다.
        """
        # 1. 이전 발언자에게 응답을 생성하고 저장
        self.response = self._generate_response()

        if self.stop:
            message = self.response
        else:
            # 2. 다음에 말할 사람을 결정
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print_w(f"\t다음 발언자: {self.next_speaker}\n")

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
    

topic = "당신들의 목표는 '온라인 채팅을 통해 고객과 소통하며 그들의 구체적인 필요를 파악하고 빠르게 구매로 이어지도록 대화를 이끌어가기 위해' 고객 유형을 빠르게 파악하고 판매 전략을 적절하게 사용하는 것입니다. 모든 과정에서 당신들의 전문 분야를 적극적으로 활용하고 서로 소통해야 합니다. 모든 과정은 문서화되어야 합니다."
director_name = "Steve"
agent_summaries = OrderedDict(
    {
        "DH": ("프로젝트 디렉터", "프로젝트 관리자"),
        "SJ": ("온라인 쇼핑몰의 챗봇에 접속한 고객이 이야기할 만한 연속된 예상 발화문 5문장을 만듭니다.", "고객의 발화문 담당자"),
        "SL": ("고객의 발화문을 확인하고 그 발화문들에서 중요한 부분만 남기도록 요약합니다.", "인텐트 요약 담당자"),
        "MS": ("고객의 발화문, 요약된 인텐트를 확인하고 고객 유형을 판단합니다. '망설이는 고객', '거부하는 고객', '의심하는 고객', 확인을 구하는 고객' 중 하나로 분류합니다.", "고객 유형 분류 담당자"),
        "SH": ("고객의 발화문, 요약된 인텐트, 고객의 유형을 확인하고 적절한 판매 전략을 구사합니다. '가벼운 대화와 지속적인 참여', '유연성과 적응성'에 집중합니다.", "판매 전략 제시 담당자"),
    }
)

agent_summary_string = "\n- ".join(
    [""]
    + [
        f"{name}: {role}, {location} 직무를 담당합니다."
        for name, (role, location) in agent_summaries.items()
    ]
)

conversation_description = f"""이것은 이 주제를 달성하기 위한 대화입니다: {topic}.
이 토론에는 온라인 비대면 세일즈 마케팅에 대해 고유한 통찰력과 전문 지식을 제공하는 {agent_summary_string}을/를 포함한 전문가 팀이 참여합니다. 고객의 예상 발화문 5문장, 인텐트 요약, 고객 유형 판단과 확신의 정도(%), 그리고 앞의 요소들을 고려한 판매 전략이 제시되어야 합니다."""
agent_descriptor_system_message = SystemMessage(
    content="각 팀원의 역할에 따른 구체적인 내용에 대해 이야기하세요."
)


def generate_agent_description(agent_name, agent_role, agent_location):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            {agent_role}업무를 하는 {agent_location} 직무 담당인 {agent_name}에 대해 전문적인 설명으로 답변해주세요. 이 때 그들의 업무와 직무를 강조해주세요.
            {agent_name}님과 직접 대화하세요.
            다른 것을 추가하지 마세요."""
        ),
    ]
    # agent_description = ChatOpenAI(model="gpt-4", temperature=1.0)(agent_specifier_prompt).content
    agent_description = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=1.0)(agent_specifier_prompt).content
    return agent_description


def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
    return f"""{conversation_description}
당신의 이름은 {agent_name}, 당신의 업무는 {agent_role}, 그리고 당신은 {agent_location}직무를 담당합니다.
당신에 대한 설명은 다음과 같습니다: {agent_description}
당신은 다음 주제에 대해 논의하고 있습니다.: {topic}
당신의 목표는 당신의 업무와 직무 담당자의 관점에서 주제에 대한 가장 유익하고 전문적이며 효과적인 대화 전략을 제공하는 것입니다.
"""


def generate_agent_system_message(agent_name, agent_header):
    return SystemMessage(
        content=(
            f"""{agent_header}
당신은 {agent_name} 스타일로 말하고 당신의 성격을 과장할 것입니다.
구체적인 결과물과 함께 이에 대한 근거를 설명해야 합니다.
같은 말을 반복해서 말하지 마세요.
{agent_name}의 관점에서 1인칭으로 말하세요.
역할을 변경하지 마세요!
다른 사람의 관점에서 말하지 마세요.
오직 {agent_name}의 관점에서만 말하세요.
당신의 관점에서 말하기가 끝나는 순간 말하기를 중단하십시오.
다른 것을 추가하지 마세요.
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
    print_w(f"\n\n{name} 설명:")
    print_w(f"\n{description}")
    print_w(f"\nHeader:\n{header}")
    print_w(f"\nSystem Message:\n{system_message.content}")

topic_specifier_prompt = [
    SystemMessage(content="당신은 이 작업을 더 구체적으로 만들 수 있습니다."),
    HumanMessage(
        content=f"""{conversation_description}
        당신의 주제에 대해 더 많이 알려주세요.
        답변이 필요한 하나의 질문으로 주제를 정리하세요.
        당신의 전문 분야 지식을 보여주세요.
        지정된 주제에 집중하세요.
        다른 것은 추가하지 마세요."""
    ),
]
# specified_topic = ChatOpenAI(model="gpt-4", temperature=1.0)(topic_specifier_prompt).content
specified_topic = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=1.0)(topic_specifier_prompt).content

print_w(f"원래 주제:\n{topic}\n")
print_w(f"구체적인 주제:\n{specified_topic}\n")

def select_next_speaker(
    step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent
) -> int:
    """
    step이 짝수이면, director를 선택합니다.
    그렇지 않으면, director가 다음 발언자를 선택합니다.
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
    # model=ChatOpenAI(model="gpt-4", temperature=0.2),
    model=ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2),
    speakers=[name for name in agent_summaries if name != director_name],
    stopping_probability=0.1,
)

agents = [director]
for name, system_message in zip(
    list(agent_summaries.keys())[1:], agent_system_messages[1:]
):
    agents.append(
        DialogueAgent(
            name=name,
            system_message=system_message,
            # model=ChatOpenAI(model="gpt-4", temperature=0.2),
            model=ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2),
        )
    )

    simulator = DialogueSimulator(
    agents=agents,
    selection_function=functools.partial(select_next_speaker, director=director),
)
simulator.reset()
simulator.inject("청중 멤버", specified_topic)
print_w(f"(청중 멤버): {specified_topic}")
print_w("\n")

while True:
    name, message = simulator.step()
    print_w(f"({name}): {message}")
    print_w("\n")
    if director.stop:
        break