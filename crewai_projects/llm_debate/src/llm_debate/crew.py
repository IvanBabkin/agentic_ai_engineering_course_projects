from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import copy
import yaml
from pathlib import Path
from .tools.custom_tool import OneTimeSearchTool

load_dotenv()

@CrewBase
class Debate():
    """Debate crew using YAML configurations with dynamic task creation"""

    def __init__(self):
        """Initialize and load YAML configurations"""
        super().__init__()
        
        # Get config directory
        config_dir = Path(__file__).parent / 'config'
        
        # Load YAML configurations
        with open(config_dir / 'agents.yaml', 'r') as file:
            self.agents_config = yaml.safe_load(file)
        
        with open(config_dir / 'tasks.yaml', 'r') as file:
            self.tasks_config = yaml.safe_load(file)

    def _create_debater_config(self, position: str) -> dict:
        """Create debater configuration with position-specific formatting"""
        base_config = copy.deepcopy(self.agents_config['debater'])
        base_config['goal'] = base_config['goal'].replace('{for/against}', position)
        return base_config

    @agent
    def debater_for(self) -> Agent:
        """Creates the 'FOR' position debater"""
        return Agent(
            config=self._create_debater_config("FOR"),
            tools=[OneTimeSearchTool()],
            verbose=True,
            memory=False,
            max_execution_time=30,
            allow_delegation=False,
            streaming=True
        )
    
    @agent
    def debater_against(self) -> Agent:
        """Creates the 'AGAINST' position debater"""
        return Agent(
            config=self._create_debater_config("AGAINST"),
            tools=[OneTimeSearchTool()],
            verbose=True,
            memory=False,
            max_execution_time=30,
            allow_delegation=False,
            streaming=True
        )

    @agent
    def judge(self) -> Agent:
        """
        Creates the impartial judge agent
        Uses its own dedicated configuration
        """
        return Agent(
            config=self.agents_config['judge'],
            verbose=True,
            memory=False,
            max_execution_time=45,  # Judges need more time to deliberate
            allow_delegation=False,
            streaming=True
        )

    def create_debate_argument_task(self, argument_num: int, position: str, context_tasks=None):
        """Create a debate argument task dynamically from YAML template"""
        
        # Determine context instruction based on whether this is an opening or response
        context_instruction = (
            "Present your opening argument." if not context_tasks else
            "Respond to your opponent's most recent argument and then present your own points."
        )
        
        # Access the loaded tasks configuration (not the file path)
        base_description = self.tasks_config['debate_argument']['description']
        
        # Format the description with our specific parameters
        description = base_description.format(
            argument_num=argument_num,
            position=position, 
            context_instruction=context_instruction,
            motion='{motion}'  # Leave this for CrewAI to handle at runtime
        )
        
        # Format the expected output
        expected_output = self.tasks_config['debate_argument']['expected_output'].format(
            argument_num=argument_num,
            position=position
        )
        
        # Select the appropriate agent
        agent = self.debater_for() if position == "FOR" else self.debater_against()
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context_tasks or []
        )

    @task
    def judge_decision(self) -> Task:
        """
        Judge decision task from YAML
        The judge evaluates all arguments and makes a final decision
        """
        return Task(
            config=self.tasks_config['judge_decision'],
            agent=self.judge()
        )

    @crew
    def crew(self) -> Crew:
        """
        Creates a debate crew with sequential execution of debate arguments.
        
        This orchestrates the entire debate flow:
        1. FOR makes opening argument
        2. AGAINST responds and counter-argues
        3. FOR responds to AGAINST and adds new points
        4. Process continues for specified number of rounds
        5. Judge evaluates and decides
        """

        num_arguments = 2  # Number of argument rounds per side
        tasks = []
        last_task = None

        # Create alternating debate rounds
        for i in range(1, num_arguments + 1):
            # FOR argument (responds to previous AGAINST argument if any)
            for_task = self.create_debate_argument_task(
                argument_num=i, 
                position="FOR", 
                context_tasks=[last_task] if last_task else []
            )
            tasks.append(for_task)
            last_task = for_task

            # AGAINST argument (always responds to the FOR argument above)
            against_task = self.create_debate_argument_task(
                argument_num=i, 
                position="AGAINST", 
                context_tasks=[last_task]  # Always has context after first FOR
            )
            tasks.append(against_task)
            last_task = against_task
        
        # Judge's decision depends on seeing all debate rounds
        judge_task = self.judge_decision()
        judge_task.context = tasks  # Judge sees the entire debate history
        
        return Crew(
            agents=[self.debater_for(), self.debater_against(), self.judge()],
            tasks=tasks + [judge_task],
            process=Process.sequential,  # One task at a time, in order
            verbose=True,
            memory=False
        )