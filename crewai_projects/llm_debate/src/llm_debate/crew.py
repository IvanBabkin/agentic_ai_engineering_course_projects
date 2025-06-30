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
            memory=True,
            max_execution_time=120
        )

    @agent
    def judge(self) -> Agent:
        """Judge agent that decides the winner"""
        return Agent(
            config=self.agents_config['judge'],
            verbose=True,
            memory=True
        )

    def create_debate_round_task(self, round_num: int, position: str, context_tasks=None):
        """Create a debate round task dynamically from YAML template"""
        
        # Set context instruction based on round
        if round_num == 1:
            context_instruction = "Present your opening argument."
        else:
            context_instruction = "Build upon previous arguments and respond to your opponent's points from earlier rounds."
        
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
            output_file=f"output/{position}_round_{round_num}.md",
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
        """Creates debate crew with dynamically generated round tasks"""
        
        tasks = []
        
        # Create 3 rounds of debate dynamically
        for round_num in range(1, 4):
            # Determine context for this round (previous tasks)
            context_tasks = tasks.copy() if round_num > 1 else []
            
            # FOR debater's turn
            for_task = self.create_debate_round_task(
                round_num=round_num, 
                position="FOR", 
                context_tasks=context_tasks
            )
            tasks.append(for_task)
            
            # AGAINST debater's turn
            against_task = self.create_debate_round_task(
                round_num=round_num, 
                position="AGAINST", 
                context_tasks=tasks.copy()  # Include the FOR task just added
            )
            tasks.append(against_task)
        
        # Add judge decision with context of all debate rounds
        judge_task = self.judge_decision()
        judge_task.context = tasks.copy()  # All debate rounds as context
        tasks.append(judge_task)
        
        # Get unique agents
        agents = [self.debater(), self.judge()]
        
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )
