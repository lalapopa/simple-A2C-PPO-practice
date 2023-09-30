import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from agent import model
from agent import dhp as DHP


def reference_state(current_state):
    y_pos = current_state[1]
    if y_pos > 0.75:
        ref_state = [0, 0.75, 0, -0.10, 0, 0, 0, 0]
    else:
        ref_state = [
            0,
            0,
            0,
            -0.025,
            0,
            0,
            0,
            0,
        ]  # x, y, Vx, Vy, phi, omega_x, leg_l, leg_r
    return ref_state


def generate_random_excitation(n_samples):
    array = []
    for i in range(0, n_samples):
        main_thrust, lat_thrust = np.random.rand(2)
        random_sign = np.power(-1, np.random.randint(1, 3))
        array.append([main_thrust * 0.1, random_sign * lat_thrust * 0.35])
    return array


def run_train(agent, ac_model):
    X, U, R, C_real = [], [], [], []
    X_pred, C_trained, action_grad, critic_grad = [], [], [], []
    list_F, list_G, list_RLS_cov, e_model = [], [], [], []
    U_ref = []
    total_steps = 1
    max_episode = 10
    max_steps = 300
    nan_occurs = False
    with tqdm(range(max_episode)) as tqdm_it:
        for i in tqdm_it:
            print(f"episode: {i}")
            # init params
            if nan_occurs:
                break
            init_condition = env.reset()[0]
            ac_model.reset()
            x = init_condition
            P = np.diag(TRACKED).astype(float)
            Q = np.diag(STATE_ERROR_WEIGHTS)
            X.append(init_condition)
            done = False
            episode_steps = 1
            while not done:
                x_ref = reference_state(x)
                R_sig = np.array(x_ref).reshape([1, -1, 1])
                U_ref.append(R_sig)
                j = 0
                x = x.reshape([1, -1, 1])
                while j < 2:
                    # Next state prediction
                    action = np.squeeze(agent.action(x, reference=R_sig))
                    action_clipped = np.clip(
                        action, np.array([0, -1.0]), np.array([1.0, 1.0])
                    )
                    x_next_pred = ac_model.predict(x, action_clipped).reshape(
                        [1, -1, 1]
                    )

                    if np.isnan(x_next_pred).any():
                        done = True
                        nan_occurs = True
                        print("=" * 8, i, j, f"in {episode_steps}", "=" * 8)
                        print(f"{x_next_pred = }\n{action =}\n{R_sig = }\n{x =}")
                        print("=" * 20)
                        break
                    # Cost prediction
                    e = np.matmul(P, x_next_pred - R_sig)
                    cost = np.matmul(np.matmul(e.transpose(0, 2, 1), Q), e)
                    dcostdx = np.matmul(2 * np.matmul(e.transpose(0, 2, 1), Q), P)

                    dactiondx = agent.gradient_actor(x, reference=R_sig)
                    lmbda = agent.value_derivative(x, reference=R_sig)

                    # Critic
                    target_lmbda = agent.target_value_derivative(
                        x_next_pred, reference=R_sig
                    )
                    A = ac_model.gradient_state(x, action)
                    B = ac_model.gradient_action(x, action)
                    # print(f"|ITER {i}||{j}| Values before grad_critic:\n{lmbda =}\n{dcostdx = }\n{agent.gamma =}\n{target_lmbda =}\n{A =}\n{B = }\n{dactiondx =}")
                    grad_critic = lmbda - np.matmul(
                        dcostdx + agent.gamma * target_lmbda,
                        A + np.matmul(B, dactiondx),
                    )
                    grad_critic = np.clip(grad_critic, -0.5, 0.5)
                    agent.update_critic(
                        x, reference=R_sig, gradient=grad_critic, learn_rate=lr_critic
                    )
                    # print(f"TOTAL GRAD CRITIC = {grad_critic}")

                    # Actor
                    lmbda = agent.value_derivative(x_next_pred, reference=R_sig)
                    grad_actor = np.matmul(dcostdx + agent.gamma * lmbda, B)
                    # print(f"TOTAL GRAD ACTOR = {grad_actor}")
                    #                grad_actor  = np.clip(grad_actor, -0.1, 0.1)
                    # grad_actor  = utils.overactuation_gradient_correction(gradients=grad_actor, actions=action, actions_clipped=action_clipped)
                    agent.update_actor(
                        x, reference=R_sig, gradient=grad_actor, learn_rate=lr_actor
                    )
                    j += 1

                X_pred.append(x_next_pred)
                C_trained.append(cost.flatten())
                action_grad.append(grad_actor)
                critic_grad.append(grad_critic)
                list_F.append(A.flatten().copy())
                list_G.append(B.flatten().copy())
                list_RLS_cov.append(ac_model.cov.copy())

                ### Run environment ###
                action = agent.action(x, reference=R_sig)
                if total_steps < 75:
                    action += excitation_signal[total_steps]
                action = np.clip(action, np.array([0, -1.0]), np.array([1.0, 1.0]))

                x_next, reward, _, _, _ = env.step(np.squeeze(action))
                print(f"next value {x_next}")

                total_steps += 1
                episode_steps += 1
                if episode_steps >= max_steps:
                    done = True

                model_error = ((x_next_pred - x_next) ** 2).mean()

                ### Real Cost ###
                e = np.matmul(P, (x_next - x_ref))
                cost = np.matmul(np.matmul(e, Q), e)

                R.append(reward)
                X.append(x_next)
                U.append(np.squeeze(action))
                e_model.append(model_error)
                C_real.append(cost)

                ### Update Model ###
                ac_model.update(x, action, x_next)

                ### Bookkeeping ###
                x = x_next
                if (
                    x_next[-1] or x_next[-2] or episode_steps >= max_steps
                ):  # break loop when legs are touching ground
                    done = True
                    print("DONE")


TENSORBOARD_DIR = "./logs/tensorboard/DHP/"

env = gym.make("LunarLander-v2", continuous=True, render_mode="human")
init_condition = env.reset()[0]
state_size = len(init_condition)
action_size = int(env.action_space.shape[0])
TRACKED = [True for i in init_condition]
excitation_signal = generate_random_excitation(75)


STATE_ERROR_WEIGHTS = [0.25, 0, 0, 1.5, 10, 1.0, 0, 0]
lr_critic = 0.01
lr_actor = 0.001
gamma_actor = 0.45

ac_kwargs = {
    # Arguments for all model types
    "state_size": state_size,
    "action_size": action_size,
    "predict_delta": False,
    # Neural Network specific args:
    "hidden_layer_size": [100, 100, 100],
    "activation": tf.nn.relu,
    # RLS specific args:
    "gamma": 0.9995,
    "covariance": 100,
    "constant": True,
    # LS specific args:
    "buffer_length": 10,
}
ac_model = model.RecursiveLeastSquares(**ac_kwargs)

kwargs = {
    "input_size": [
        state_size,
        state_size,
    ],  # [Aircraft state size, Number of tracked states]
    "output_size": action_size,  # Actor output size (Critic output is dependend only on aircraft state size)
    "hidden_layer_size": [
        100,
        100,
        100,
    ],  # List with number of nodes per layer, number of layers is variable
    "kernel_stddev": 0.1,  # Standard deviation used in the truncated normal distribution to initialize all parameters
    "lr_critic": lr_critic,  # Learn rate Critic
    "lr_actor": lr_actor,  # Learn rate Actor
    "gamma": gamma_actor,  # Discount factor
    "use_bias": False,  # Use bias terms (only if you train using batched data)
    "split": False,  # Split architechture of the actor, if False, a single fully connected layer is used.
    "target_network": False,  # Use target networks
    "activation": tf.keras.layers.Activation("relu"),
    "log_dir": TENSORBOARD_DIR,  # Where to save checkpoints
    "use_delta": (
        True,
        TRACKED,
    ),  # (True, TRACKED) used 's = [x, (x - x_ref)]' || (False, None) uses 's = [x, x_ref]'
}
agent = DHP.Agent(**kwargs)


cost_step = sum(C_real) / total_steps
print(f"Deluted cost = {cost_step}, num steps {total_steps}")

fig, axis = plt.subplots(4, 3, figsize=(14, 10))
axis[0, 0].plot([i[0] for i in X], label="X position")
axis[0, 0].plot([i[1] for i in X], label="Y position")
# axis[0, 0].plot([np.squeeze(i)[0] for i in X_pred], label="X position (pred)")
axis[0, 0].plot([np.squeeze(i)[0] for i in U_ref], "--", label="X position (ref)")
# axis[0, 0].plot([np.squeeze(i)[1] for i in X_pred], label="Y position (pred)")
axis[0, 0].legend()

axis[0, 1].plot([i[2] for i in X], label="V_x speed")
axis[0, 1].plot([i[3] for i in X], label="V_y speed")
axis[0, 1].plot([np.squeeze(i)[3] for i in U_ref], "--", label="V_y speed (ref)")
axis[0, 1].legend()

axis[0, 2].plot([i[4] for i in X], label="angle")
axis[0, 2].legend()

axis[1, 2].plot([i[5] for i in X], label="angular velocity")
axis[1, 2].legend()

axis[1, 0].plot([i[6] for i in X], label="left leg")
axis[1, 0].plot([i[7] for i in X], label="right leg")
axis[1, 0].legend()

axis[1, 1].plot([i[0] for i in U], label="main thrust")
axis[1, 1].plot([i[1] for i in U], label="lat thrust")
axis[1, 1].legend()

axis[2, 0].plot(C_real, label="cost_real")
axis[2, 0].plot(C_trained, label="cost_predicted")
axis[2, 0].legend()

axis[2, 1].plot([np.squeeze(i) for i in critic_grad], label="critic grad")
axis[2, 1].legend()
axis[3, 1].plot([np.squeeze(i) for i in action_grad], label="actor grad")
axis[3, 1].legend()

axis[2, 2].plot(R, label="Reward")
axis[2, 2].legend()

plt.savefig("result.png")
