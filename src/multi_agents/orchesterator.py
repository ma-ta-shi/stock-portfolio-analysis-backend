from agentA import AgentA 
from agentB import AgentB
from tools import run_shell, list_files, read_file
from rich import print

agentA = AgentA()
agentB = AgentB()

def multi_agent_task(user_prompt):
    # Step 1: manager decides task
    manager_instruction = agentB.run(
        f"""
        User asked:
        {user_prompt}

        Create instructions for worker.
        """
    )

    print("\n[cyan]Manager Instructions:[/cyan]")
    print(manager_instruction)

    # Step 2: worker executes
    shell_result = run_shell("pwd && ls -la")

    worker_output = agentA.run(
        f"""
        Task from manager:
        {manager_instruction}

        Tool output:
        {shell_result}

        Complete task.
        """
    )
    print(agentA.message)
    print("\n[green]Worker Output:[/green]")
    print(worker_output)

    # Step 3: manager reviews worker
    final_answer = agentB.run(
        f"""
        Review worker output.

        Worker said:
        {worker_output}

        Improve/correct if needed.
        Return final answer.
        """
    )
    print(agentB.message)
    return final_answer


if __name__ == "__main__":
    while True:
        user = input("\nYou> ")

        if user.lower() == "exit":
            break

        result = multi_agent_task(user)

        print("\n[bold yellow]Final Answer:[/bold yellow]")
        print(result)