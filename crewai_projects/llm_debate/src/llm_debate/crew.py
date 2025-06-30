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

    def create_debate_argument_task(self, argument_num: int, position: str, context_tasks=None):
        """Create a debate argument task dynamically from YAML template"""
        
        # Set context instruction based on whether there's a preceding argument
        if not context_tasks:
            context_instruction = "Present your opening argument."
        else:
            context_instruction = "Respond to your opponent's most recent argument and then present your own points."
        
        # Create description by substituting only our custom placeholders
        # Leave {motion} for CrewAI to resolve from inputs
        base_description = self.tasks_config['debate_argument']['description']
        description = base_description.replace('{argument_num}', str(argument_num)).replace('{position}', position).replace('{context_instruction}', context_instruction)
        
        # Create expected output
        base_expected_output = self.tasks_config['debate_argument']['expected_output']
        expected_output = base_expected_output.replace('{argument_num}', str(argument_num)).replace('{position}', position)
        
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
        """Creates a debate crew with sequential execution of debate arguments."""
        
        num_arguments = 3
        tasks = []
        last_task = None

        for i in range(1, num_arguments + 1):
            # FOR argument
            for_task = self.create_debate_argument_task(
                argument_num=i, 
                position="FOR", 
                context_tasks=[last_task] if last_task else []
            )
            tasks.append(for_task)
            last_task = for_task

            # AGAINST argument
            against_task = self.create_debate_argument_task(
                argument_num=i, 
                position="AGAINST", 
                context_tasks=[last_task]
            )
            tasks.append(against_task)
            last_task = against_task
        
        # Judge's decision (depends on all debate rounds)
        judge_task = self.judge_decision()
        judge_task.context = tasks
        
        return Crew(
            agents=[self.debater(), self.judge()],
            tasks=tasks + [judge_task],
            process=Process.sequential,
            verbose=True,
            memory=False,
        )
