from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv

load_dotenv()

@CrewBase
class Debate():
    """Debate crew using YAML configurations with dynamic task creation"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def debater(self) -> Agent:
        """Debater agent that can argue either side"""
        return Agent(
            config=self.agents_config['debater'],
            verbose=True,
            memory=False,
            max_execution_time=30,
            allow_delegation=False,
            streaming=True
        )

    @agent
    def judge(self) -> Agent:
        """Judge agent that decides the winner"""
        return Agent(
            config=self.agents_config['judge'],
            verbose=True,
            memory=False,
            max_execution_time=45,
            allow_delegation=False,
            streaming=True
        )

    def create_debate_round_task(self, round_num: int, position: str, context_tasks=None):
        """Create a debate round task dynamically from YAML template"""
        
        # Set context instruction based on whether there's a preceding argument
        if not context_tasks:
            context_instruction = "Present your opening argument."
        else:
            context_instruction = "Respond to your opponent's most recent argument and then present your own points."
        
        # Create description by substituting only our custom placeholders
        # Leave {motion} for CrewAI to resolve from inputs
        base_description = self.tasks_config['debate_round']['description']
        description = base_description.replace('{round_num}', str(round_num)).replace('{position}', position).replace('{context_instruction}', context_instruction)
        
        # Create expected output
        base_expected_output = self.tasks_config['debate_round']['expected_output']
        expected_output = base_expected_output.replace('{round_num}', str(round_num)).replace('{position}', position)
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.debater(),
            context=context_tasks or []
        )

    @task
    def judge_decision(self) -> Task:
        """Judge decision task from YAML"""
        return Task(
            config=self.tasks_config['judge_decision'],
            agent=self.judge()
        )

    @crew
    def crew(self) -> Crew:
        """Creates a debate crew with sequential execution of debate rounds."""
        
        # Round 1
        for_task_r1 = self.create_debate_round_task(round_num=1, position="FOR")
        against_task_r1 = self.create_debate_round_task(
            round_num=1, position="AGAINST", context_tasks=[for_task_r1]
        )
        
        # Round 2
        for_task_r2 = self.create_debate_round_task(
            round_num=2, position="FOR", context_tasks=[against_task_r1]
        )
        against_task_r2 = self.create_debate_round_task(
            round_num=2, position="AGAINST", context_tasks=[for_task_r2]
        )
        
        # Round 3
        for_task_r3 = self.create_debate_round_task(
            round_num=3, position="FOR", context_tasks=[against_task_r2]
        )
        against_task_r3 = self.create_debate_round_task(
            round_num=3, position="AGAINST", context_tasks=[for_task_r3]
        )
        
        # Judge's decision (depends on all debate rounds)
        judge_task = self.judge_decision()
        all_debate_tasks = [
            for_task_r1, against_task_r1,
            for_task_r2, against_task_r2,
            for_task_r3, against_task_r3,
        ]
        judge_task.context = all_debate_tasks
        
        return Crew(
            agents=[self.debater(), self.judge()],
            tasks=all_debate_tasks + [judge_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
        )
