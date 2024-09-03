"""
 Copyright (c) 2024, Your Name
 All rights reserved.
 CopyrightText: SEO HYEON HO
 License-Identifier: Apache-2.0

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List
import time 

WIDTH, HEIGHT, DEPTH = 110, 110, 110

class Agent:
    def __init__(self, start, goal, waypoint_param, radius=5, agent_id=None, color=None):    
        self.position = np.array(start, dtype=np.float64)
        self.final_goal = np.array(goal, dtype=np.float64)
        self.waypoint_param = waypoint_param
        self.waypoints = self.generate_waypoints(start, goal)
        self.current_waypoint_index = 0
        self.goal = self.waypoints[self.current_waypoint_index]
        self.velocity = np.array([0, 0, 0], dtype=np.float64)
        self.acceleration = np.array([0, 0, 0], dtype=np.float64)
        self.max_speed = 2
        self.max_force = 10
        self.arrival_radius = 1
        self.radius = radius
        self.agent_id = agent_id
        self.color = color
        self.path = [np.array(start, dtype=np.float64)]  
        self.pause_time = 0
        self.reached_goal = False  
        
    def generate_waypoints(self, start, goal):
        midpoint = start
        return [midpoint, goal]
    
    def apply_force(self, force):
        self.acceleration += force

    def update(self, agents):
        if self.pause_time > 0:
            self.pause_time -= 1
            return
        
        distance_to_current_goal = np.linalg.norm(self.goal - self.position)

        if distance_to_current_goal < self.arrival_radius:
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(self.waypoints):
                self.goal = self.waypoints[self.current_waypoint_index]
            else:
                self.goal = self.final_goal
                self.reached_goal = True 

        desired = self.goal - self.position
        distance = np.linalg.norm(desired)
        if distance < self.arrival_radius:
            desired = np.array([0, 0, 0], dtype=np.float64)
        else:
            desired = self.limit(desired, self.max_speed)
        seek_force = desired - self.velocity
        seek_force = self.limit(seek_force, self.max_force)
        self.apply_force(seek_force)

        separation = np.zeros(3)
        for other in agents:
            if other is not self:
                diff = self.position - other.position
                distance = np.linalg.norm(diff)
                if distance < 1.7*(self.radius + other.radius):
                    separation += diff
                    if distance < (self.radius + other.radius):
                        print(f'충돌: {self.agent_id}번과 {other.agent_id}번 기체')
        separation_force = self.limit(separation, self.max_force)
        self.apply_force(separation_force)

        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, self.max_speed)
        self.position += self.velocity
        self.acceleration *= 0
        self.path.append(np.copy(self.position))  

    def pause(self, duration):
        self.pause_time = duration

    def limit(self, vector, max_value):
        if np.linalg.norm(vector) > max_value:
            return vector / np.linalg.norm(vector) * max_value
        return vector

def check_collisions(agents):
    collided_agents = []
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i+1:], start=i+1):
            diff = agent1.position - agent2.position
            distance = np.linalg.norm(diff)
            if distance < (agent1.radius + agent2.radius):
                print(f'충돌: {agent1.agent_id}번과 {agent2.agent_id}번 기체')
                collided_agents.extend([agent1, agent2])
    
    if collided_agents:
        farthest_agent = max(collided_agents, key=lambda a: np.linalg.norm(a.goal - a.position))
        for agent in collided_agents:
            if agent != farthest_agent:
                agent.pause(15)

def update(frame, ax, agents):
    ax.clear()
    for agent in agents:
        agent.update(agents)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        display_radius = agent.radius * 1
        x = display_radius * np.cos(u) * np.sin(v) + agent.position[0]
        y = display_radius * np.sin(u) * np.sin(v) + agent.position[1]
        z = display_radius * np.cos(v) + agent.position[2]
        ax.plot_surface(x, y, z, color=agent.color, alpha=0.8)
        
        ax.scatter(agent.goal[0], agent.goal[1], agent.goal[2], c='r', s=40)

        path = np.array(agent.path)
        ax.scatter(path[:, 0], path[:, 1], path[:, 2], c=agent.color, s=1, alpha=0.5)

    check_collisions(agents)
    ax.set_xlim(-WIDTH, WIDTH)
    ax.set_ylim(-HEIGHT, HEIGHT)
    ax.set_zlim(-DEPTH, DEPTH)
    ax.set_title("3D Agents Pathfinding Simulation")
    if all_agents_reached_goal(agents):
        end_time = time.time()
        print(f"총 소요 시간: {end_time - start_time:.2f}초")
        return None  
    
    return ax

def all_agents_reached_goal(agents):
    return all(agent.reached_goal for agent in agents)

def setup_scenario() -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    num_agents = 64
    radius = 100

    indices = np.arange(num_agents)
    theta = np.e * (3 - np.sqrt(5)) * indices 
    phi = np.arccos(1 - 2 * (indices + 0.5) / num_agents)  

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    agents = np.column_stack((x, y, z))

    goals = -agents
    colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
    return agents, goals, colors

# def setup_scenario() -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
#     num_agents = 7  
#     radius = 100

#     angles = np.linspace(0, 2 * np.e, num_agents, endpoint=False) 
#     x = radius * np.cos(angles) 
#     y = radius * np.sin(angles) 
#     z = np.zeros(num_agents)     

#     start_points = np.column_stack((x, y, z))
#     goal_points = -start_points
#     print(f'start:{start_points}, goal_points:{goal_points}')

#     colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
#     return start_points, goal_points, colors

def main():
    global start_time
    start_time = time.time()
    start_points, goal_points, colors = setup_scenario()
    agents = [Agent(start, goal, waypoint_param=start, radius=3, agent_id=i, color=colors[i]) for i, (start, goal) in enumerate(zip(start_points, goal_points))]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_zlim(-110, 110)

    anim = FuncAnimation(fig, update, frames=200, fargs=(ax, agents), interval=50)

    plt.show()   

if __name__ == "__main__":
    main()
