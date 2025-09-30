from openai import OpenAI
import os
import sys
import csv
from datetime import datetime
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")


def get_completion(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    content = completion.choices[0].message.content
    return content

class Agent:
    def __init__(self, name, expertise, original_team_id=None):
        self.name = name
        self.expertise = expertise
        self.persona = None
        self.memory = []  # Store full conversation history of teams the agent was part of
        self.original_team_id = original_team_id  # Track original team to prevent reassignment

    def generate_persona(self, problem, team_id=None, num_teams=None):
        prompt = f"Generate a detailed persona for an agent named {self.name} with expertise in {self.expertise} to solve the problem: '{problem}'. "
        if team_id and num_teams:
            prompt += f"This agent is part of Team {team_id} out of {num_teams} teams working on this problem, so create a unique background, personality, and skill set that would provide a distinct perspective from agents in other teams. "
        prompt += "Include background, personality, and specific skills. Keep under 200 words."
        self.persona = get_completion(prompt).strip()

    def respond(self, solution_history, prompt, round_num, team_id):
        full_prompt = f"You are {self.name}, with this persona: {self.persona}. "
        full_prompt += f"Your expertise is {self.expertise}. "
        full_prompt += f"Problem: {problem}\n\n"
        full_prompt += f"You are in Round {round_num}, Team {team_id}.\n"
        if round_num > 1:
        	full_prompt += "The composition of the team is now different from Round 1."
        full_prompt += "Previous conversations you were part of:\n" + "\n".join(self.memory) + "\n\n" if self.memory else "You have no previous conversation history.\n\n"
        full_prompt += "Current team discussion:\n" + "\n".join(solution_history) + "\n\n" if solution_history else "No current team discussion yet.\n\n"
        full_prompt += prompt + "\n\nRespond naturally as in a dialogue or conversation with your colleagues. Keep your response concise, under 300 words. Write in full paragraphs - do NOT use bullet points, numbered lists, or any list formatting. The discussion should take the form of a debate, but you can also agree to points that are largely compatible with your perspective. Reference specific ideas from previous speakers in your current team or from your own past team conversations (stored in your memory), but do NOT reference discussions from teams you were not part of."
        response = get_completion(full_prompt).strip()
        return response

class JudgeAgent:
    def __init__(self, name="Judge", expertise="solution synthesis and evaluation"):
        self.name = name
        self.expertise = expertise
        self.persona = None
        self.memory = []

    def generate_persona(self, problem):
        prompt = f"Generate a detailed persona for an agent named {self.name} with expertise in {self.expertise} to evaluate solutions for the problem: '{problem}'. "
        prompt += "Include background, personality, and skills in critical analysis and decision-making. Keep under 200 words."
        self.persona = get_completion(prompt).strip()

    def evaluate_solutions(self, all_team_summaries, problem):
        full_prompt = f"You are {self.name}, with this persona: {self.persona}. "
        full_prompt += f"Your expertise is {self.expertise}. "
        full_prompt += f"Problem: {problem}\n\n"
        
        for team_idx, team_summary in enumerate(all_team_summaries, 1):
            full_prompt += f"Team {team_idx} Summary:\n{team_summary}\n\n"
        
        full_prompt += "Review all team summaries, evaluate their strengths and weaknesses, and compose a final, comprehensive solution that integrates the best ideas. Write this as a formal letter to decision makers who do not have access to the team discussions. The letter should:\n"
        full_prompt += "- Be written entirely in full, well-structured paragraphs (absolutely NO bullet points, numbered lists, or any list formatting)\n"
        full_prompt += "- Be comprehensive and descriptive of the plan in its details\n"
        full_prompt += "- Fill in any gaps where necessary\n"
        full_prompt += "- Target a length of 2000 words\n"
        full_prompt += "- If the problem requires code, include the complete code after the letter (in addition to the 2000 word letter)\n"
        full_prompt += "Remember: Write only in prose format with full paragraphs. The letter should read smoothly as a continuous narrative without any lists or bullet points."
        final_solution = get_completion(full_prompt).strip()
        self.memory.append(f"Final Solution for problem '{problem}': {final_solution}")
        return final_solution

    def interact_with_user(self, all_team_summaries, problem):
        self.memory.extend([f"Team {i+1} Summary: {summary}" for i, summary in enumerate(all_team_summaries)])
        print(f"\n{'='*60}")
        print("INTERACTIVE DIALOGUE WITH JUDGE")
        print(f"{'='*60}\n")
        print(f"You can now ask {self.name} questions about the final decision or the team summaries.\n")

        while True:
            user_input = input("Your question (or type 'exit' to end): ").strip()
            if user_input.lower() == 'exit':
                break
            prompt = f"You are {self.name}, with this persona: {self.persona}. "
            prompt += f"Your expertise is {self.expertise}. "
            prompt += f"You have just evaluated solutions for the problem: '{problem}'. "
            prompt += "Here is the relevant information you have:\n"
            for memory_item in self.memory:
                prompt += f"{memory_item}\n\n"
            prompt += f"User question: {user_input}\n\n"
            prompt += "Addressing the user's question directly while referencing relevant parts of the team summaries or final solution as needed."
            response = get_completion(prompt).strip()
            print(f"\n{self.name} ({self.expertise}): {response}\n")
            self.memory.append(f"User asked: {user_input}\nJudge responded: {response}")

class ModeratorAgent:
    def __init__(self, name="Moderator", expertise="discussion synthesis and summary"):
        self.name = name
        self.expertise = expertise
        self.persona = None

    def generate_persona(self, problem):
        prompt = f"Generate a detailed persona for an agent named {self.name} with expertise in {self.expertise} to synthesize team discussions for the problem: '{problem}'. "
        prompt += "Include background, personality, and skills in extracting key insights, identifying relevant information, and creating coherent summaries. Keep under 200 words."
        self.persona = get_completion(prompt).strip()

    def synthesize_discussion(self, team_id, team_solutions, problem):
        full_prompt = f"You are {self.name}, with this persona: {self.persona}. "
        full_prompt += f"Your expertise is {self.expertise}. "
        full_prompt += f"Problem: {problem}\n\n"
        full_prompt += f"Team {team_id} Discussion:\n" + "\n".join(team_solutions) + "\n\n"
        full_prompt += "Synthesize this team's discussion into a coherent summary that:\n"
        full_prompt += "- Captures the key ideas, proposals, and insights from the discussion\n"
        full_prompt += "- Removes irrelevant tangents, repetitions, and off-topic elements\n"
        full_prompt += "- Highlights areas of agreement and key points of debate\n"
        full_prompt += "- Preserves the essential reasoning and evidence presented\n"
        full_prompt += "- Is written entirely in full, well-structured paragraphs (absolutely NO bullet points, numbered lists, or any list formatting)\n"
        full_prompt += "- Target a length of 1000 words\n"
        full_prompt += "Remember: Write only in prose format with full paragraphs as a flowing narrative summary."
        return get_completion(full_prompt).strip()

def get_num_agents():
    while True:
        try:
            num = int(input("How many collaborating agents should participate per team (excluding Judge)? (2-5): "))
            if 2 <= num <= 5:
                return num
            print("Please enter a number between 2 and 5.")
        except ValueError:
            print("Please enter a valid integer.")

def get_num_teams():
    while True:
        try:
            num = int(input("How many independent teams should work on the problem? (1-10): "))
            if 1 <= num <= 10:
                return num
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer.")

def get_num_agents_to_swap(num_agents, num_teams):
    max_swap = min(num_agents - 1, num_teams - 1)
    while True:
        try:
            num = int(input(f"How many agents should swap teams for the second round? (0-{max_swap}): "))
            if 0 <= num <= max_swap:
                return num
            print(f"Please enter a number between 0 and {max_swap}.")
        except ValueError:
            print("Please enter a valid integer.")

def generate_relevant_expertises(problem, num_agents, team_id, num_teams):
    prompt = f"Given the problem: '{problem}', generate a list of {num_agents} distinct areas of expertise relevant to solving this problem. "
    prompt += f"This is for Team {team_id} out of {num_teams} teams, so provide a unique perspective and approach different from other teams. "
    prompt += "Each expertise should be a concise phrase (e.g., 'environmental science', 'public health'). Return the list as a comma-separated string."
    expertises = get_completion(prompt).strip().split(',')
    expertises = [exp.strip() for exp in expertises]
    
    while len(expertises) < num_agents:
        expertises.append(f"specialized expertise {len(expertises) + 1}")
    return expertises[:num_agents]

def get_agent_expertises(num_agents, problem, team_id, num_teams):
    expertises = generate_relevant_expertises(problem, num_agents, team_id, num_teams)
    agents_config = []
    for i in range(num_agents):
        name = f"Agent{i+1}"
        expertise = expertises[i]
        agents_config.append((name, expertise, team_id))
    return agents_config

def save_solutions_to_csv(problem, all_teams_agents, moderator, judge, all_team_solutions, all_team_summaries, final_solution, second_round_teams=None, second_round_solutions=None, second_round_summaries=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"solutions_{problem[:40].replace(' ', '_')}_{timestamp}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Team', 'Agent', 'Expertise', 'Persona', 'Contribution'])
        
        # Write first-round team agents
        for team_idx, agents in enumerate(all_teams_agents, 1):
            for agent in agents:
                persona_safe = agent.persona.replace('\n', '\\n')
                writer.writerow([f"Team{team_idx}", agent.name, agent.expertise, persona_safe, ""])
        
        # Write moderator and judge personas
        persona_safe = moderator.persona.replace('\n', '\\n')
        writer.writerow(["Moderator", moderator.name, moderator.expertise, persona_safe, ""])
        persona_safe = judge.persona.replace('\n', '\\n')
        writer.writerow(["Judge", judge.name, judge.expertise, persona_safe, ""])
        
        # Write first-round team contributions
        for team_idx, team_solutions in enumerate(all_team_solutions, 1):
            for contribution in team_solutions:
                agent_name, text = contribution.split(": ", 1)
                text_safe = text.replace('\n', '\\n')
                writer.writerow([f"Team{team_idx}", agent_name, "", "", text_safe])
        
        # Write first-round summaries
        for team_idx, team_summary in enumerate(all_team_summaries[:len(all_team_solutions)], 1):
            summary_safe = team_summary.replace('\n', '\\n')
            writer.writerow([f"Team{team_idx}_Summary", moderator.name, "", "", summary_safe])
        
        # Write second-round teams and contributions if applicable
        if second_round_teams and second_round_solutions and second_round_summaries:
            for team_idx, agents in enumerate(second_round_teams, 1):
                for agent in agents:
                    writer.writerow([f"SecondRound_Team{team_idx}", agent.name, agent.expertise, agent.persona.replace('\n', '\\n'), ""])
            
            for team_idx, team_solutions in enumerate(second_round_solutions, 1):
                for contribution in team_solutions:
                    agent_name, text = contribution.split(": ", 1)
                    text_safe = text.replace('\n', '\\n')
                    writer.writerow([f"SecondRound_Team{team_idx}", agent_name, "", "", text_safe])
            
            # Write second-round summaries
            for team_idx, team_summary in enumerate(second_round_summaries, 1):
                summary_safe = team_summary.replace('\n', '\\n')
                writer.writerow([f"SecondRound_Team{team_idx}_Summary", moderator.name, "", "", summary_safe])
        
        # Write judge's final solution
        final_solution_safe = final_solution.replace('\n', '\\n')
        writer.writerow(["Judge", judge.name, "", "", final_solution_safe])
    
    print(f"\nSolutions saved to {filename}")

def run_team_solving(team_id, agents, problem, turns_per_agent, round_name="First Round", round_num=1):
    solution_history = []
    
    print(f"\n{'='*60}")
    print(f"{round_name.upper()} - TEAM {team_id}")
    print(f"{'='*60}\n")

    for turn_num in range(1, turns_per_agent + 1):
        print(f"Team {team_id} - Turn {turn_num}:\n")

        for agent in agents:
            if turn_num == 1:
                agent_prompt = f"Problem: {problem}\n\nAs the discussion begins, share your initial perspective and proposed approach based on your expertise."
            else:
                agent_prompt = "Continue the discussion by responding to your colleagues' points, refining ideas, building on good suggestions, or raising concerns based on your expertise."
            
            response = agent.respond(solution_history, agent_prompt, round_num, team_id)
            print(f"{agent.name} ({agent.expertise}): {response}\n")
            solution_history.append(f"{agent.name}: {response}")
    
    # Store the full team conversation in each agent's memory
    for agent in agents:
        agent.memory.append(f"Round {round_num}, Team {team_id} Conversation:\n" + "\n".join(solution_history))
    
    return solution_history

def swap_agents(all_teams_agents, num_agents_to_swap, num_agents):
    if len(all_teams_agents) < 2:
        print("Not enough teams to swap agents.")
        return all_teams_agents
    
    if num_agents_to_swap == 0:
        print("No agents selected for swapping. Proceeding with original teams for second round.")
        return all_teams_agents
    
    # Ensure num_agents_to_swap is feasible (each team swaps to num_teams-1 other teams)
    if num_agents_to_swap > num_agents:
        print(f"Cannot swap {num_agents_to_swap} agents; max is {num_agents}. Setting to {num_agents}.")
        num_agents_to_swap = num_agents
    if num_agents_to_swap > len(all_teams_agents) - 1:
        print(f"Cannot swap {num_agents_to_swap} agents to {len(all_teams_agents)-1} other teams. Setting to {len(all_teams_agents)-1}.")
        num_agents_to_swap = len(all_teams_agents) - 1
    
    num_teams = len(all_teams_agents)
    new_teams = [[] for _ in range(num_teams)]
    remaining_agents = [team[:] for team in all_teams_agents]
    
    # Step 1: Each team contributes num_agents_to_swap agents to a pool
    swap_pool = []
    for team_idx, team in enumerate(all_teams_agents):
        if len(team) < num_agents_to_swap:
            print(f"Error: Team {team_idx + 1} has only {len(team)} agents, cannot swap {num_agents_to_swap}.")
            return all_teams_agents
        selected_agents = random.sample(team, num_agents_to_swap)
        swap_pool.extend([(agent, team_idx) for agent in selected_agents])
        remaining_agents[team_idx] = [agent for agent in team if agent not in selected_agents]
    
    # Step 2: Distribute agents from each team to different other teams
    for source_team_idx in range(num_teams):
        # Get agents from this team in the swap pool
        team_agents = [(agent, orig_team) for agent, orig_team in swap_pool if orig_team == source_team_idx]
        if len(team_agents) != num_agents_to_swap:
            print(f"Error: Team {source_team_idx + 1} has {len(team_agents)} agents in pool, expected {num_agents_to_swap}.")
            return all_teams_agents
        
        # Get possible target teams (exclude source team)
        target_teams = [i for i in range(num_teams) if i != source_team_idx]
        if len(target_teams) < num_agents_to_swap:
            print(f"Error: Not enough target teams for Team {source_team_idx + 1} to swap {num_agents_to_swap} agents.")
            return all_teams_agents
        
        # Assign each agent to a different target team
        random.shuffle(target_teams)
        for (agent, _), target_team_idx in zip(team_agents, target_teams[:num_agents_to_swap]):
            new_teams[target_team_idx].append(agent)
            swap_pool.remove((agent, source_team_idx))
    
    # Step 3: Distribute remaining agents to maintain num_agents per team
    for team_idx, team in enumerate(remaining_agents):
        new_teams[team_idx].extend(team)
    
    # Step 4: Verify and adjust team sizes
    for team_idx, team in enumerate(new_teams):
        while len(team) < num_agents:
            print(f"Warning: Team {team_idx + 1} has {len(team)} agents, adding an available agent...")
            # Find an agent from another team with more than num_agents, not from this team's original agents
            for other_team_idx, other_team in enumerate(new_teams):
                if other_team_idx != team_idx and len(other_team) > num_agents:
                    valid_agents = [agent for agent in other_team if agent.original_team_id != team_idx]
                    if valid_agents:
                        agent = random.choice(valid_agents)
                        other_team.remove(agent)
                        new_teams[team_idx].append(agent)
                        break
        while len(team) > num_agents:
            print(f"Warning: Team {team_idx + 1} has {len(team)} agents, removing an agent...")
            agent = random.choice(team)
            team.remove(agent)
            # Find a team with fewer than num_agents, not agent's original team
            for other_team_idx, other_team in enumerate(new_teams):
                if other_team_idx != team_idx and len(other_team) < num_agents and agent.original_team_id != other_team_idx:
                    other_team.append(agent)
                    break
    
    # Final verification
    for team_idx, team in enumerate(new_teams):
        if len(team) != num_agents:
            print(f"Error: Team {team_idx + 1} has {len(team)} agents, expected {num_agents}. Reverting to original teams.")
            return all_teams_agents
    
    return new_teams

def multi_agent_problem_solving(problem, turns_per_agent=3):
    num_teams = get_num_teams()
    num_agents = get_num_agents()

    if num_teams > 1:    
            num_agents_to_swap = get_num_agents_to_swap(num_agents, num_teams)
    
    all_teams_agents = []
    all_team_solutions = []
    all_team_summaries = []
    
    print("\nGenerating Teams and Personas:\n")
    
    for team_id in range(1, num_teams + 1):
        print(f"\n--- Team {team_id} ---")
        agents_config = get_agent_expertises(num_agents, problem, team_id, num_teams)
        
        agents = [Agent(f"T{team_id}_{name}", expertise, team_id) for name, expertise, _ in agents_config]
        
        for agent in agents:
            agent.generate_persona(problem, team_id, num_teams)
            print(f"Persona for {agent.name} ({agent.expertise}):\n{agent.persona}\n")
        
        all_teams_agents.append(agents)
    
    moderator = ModeratorAgent()
    moderator.generate_persona(problem)
    print(f"\nPersona for {moderator.name} ({moderator.expertise}):\n{moderator.persona}\n")
    
    judge = JudgeAgent()
    judge.generate_persona(problem)
    print(f"\nPersona for {judge.name} ({judge.expertise}):\n{judge.persona}\n")

    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print(f"{'='*60}\n")

    for team_id, agents in enumerate(all_teams_agents, 1):
        team_solutions = run_team_solving(team_id, agents, problem, turns_per_agent, "First Round", round_num=1)
        all_team_solutions.append(team_solutions)
        
        print(f"\n{'='*60}")
        print(f"MODERATOR'S SYNTHESIS OF TEAM {team_id} (First Round)")
        print(f"{'='*60}\n")
        team_summary = moderator.synthesize_discussion(team_id, team_solutions, problem)
        print(f"{moderator.name}: {team_summary}\n")
        all_team_summaries.append(team_summary)

    second_round_teams = None
    second_round_solutions = None
    second_round_summaries = None

    if num_teams > 1:
        print(f"\n{'='*60}")
        print("SECOND ROUND SETUP")
        print(f"{'='*60}\n")
        second_round_teams = swap_agents(all_teams_agents, num_agents_to_swap, num_agents)
        
        second_round_solutions = []
        second_round_summaries = []
        
        for team_id, agents in enumerate(second_round_teams, 1):
            print(f"\nSecond Round - Team {team_id} Agents:")
            for agent in agents:
                print(f"- {agent.name} ({agent.expertise})")
            
            team_solutions = run_team_solving(team_id, agents, problem, turns_per_agent, "Second Round", round_num=2)
            second_round_solutions.append(team_solutions)
            
            print(f"\n{'='*60}")
            print(f"MODERATOR'S SYNTHESIS OF TEAM {team_id} (Second Round)")
            print(f"{'='*60}\n")
            team_summary = moderator.synthesize_discussion(team_id, team_solutions, problem)
            print(f"{moderator.name}: {team_summary}\n")
            second_round_summaries.append(team_summary)
        
        all_team_summaries.extend(second_round_summaries)

    print(f"\n{'='*60}")
    print("JUDGE'S FINAL SOLUTION (Integrating All Teams)")
    print(f"{'='*60}\n")
    final_solution = judge.evaluate_solutions(all_team_summaries, problem)
    print(f"{judge.name} ({judge.expertise}): {final_solution}\n")

    save_solutions_to_csv(problem, all_teams_agents, moderator, judge, all_team_solutions, all_team_summaries, final_solution, second_round_teams, second_round_solutions, second_round_summaries)
    
    judge.interact_with_user(all_team_summaries, problem)
    
    print("\nProblem-solving concluded.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        problem = " ".join(sys.argv[1:])
    else:
        problem = "How to reduce bisphenol use in receipts?"
    multi_agent_problem_solving(problem)
