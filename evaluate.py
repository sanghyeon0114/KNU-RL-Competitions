import gymnasium as gym

from agent import DoubleDQNAgent

BEST_PATH = "double_dqn_best.pkl"

def evaluate(path: str = BEST_PATH, seed: int | None = None):
    env = gym.make('kymnasium/AvoidBlurp-Discrete-Ballistic-Normal-Stage-1', render_mode='human', bgm=True)
    agent = DoubleDQNAgent.load(path)

    times = []
    
    obs, info = env.reset(seed=seed) 
    done = False
    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    times.append(info['time_elapsed'])
    print(f"Seed: {seed}, Time: {info['time_elapsed']:.2f}, Result: {info['game_state']}")

    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, nargs="?", default=None)
    parser.add_argument("--model", default=BEST_PATH)
    args = parser.parse_args()
    evaluate(args.model, args.seed)
