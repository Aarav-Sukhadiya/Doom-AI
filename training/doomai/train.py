"""Training entry point: python -m doomai.train <same flags as doomai.ppo>.

Worker processes re-import the parent's __main__ module, so the launcher stays free of torch and
imports the trainer only inside run(); each game worker then holds tens of MB instead of ~320.
"""


def run() -> None:
    from doomai.ppo import main

    main()


if __name__ == "__main__":
    run()
