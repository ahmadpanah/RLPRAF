import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import defaultdict

class RLPRAF:
    def __init__(self):
        # Initialize states and actions
        self.states = ['normal', 'underutilized', 'overutilized']
        self.actions = ['no_operation', 'scale_up', 'scale_down']
        
        # Q-learning parameters
        self.q_table = defaultdict(lambda: {action: 0.0 for action in self.actions})
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # VM configuration
        self.vm_configs = {
            'small': {'cpu': 1, 'ram': 2, 'cost': 0.1},
            'medium': {'cpu': 2, 'ram': 4, 'cost': 0.2},
            'large': {'cpu': 4, 'ram': 8, 'cost': 0.4}
        }
        
        # Thresholds
        self.lower_threshold = 0.3
        self.upper_threshold = 0.8
        
        # Current system state
        self.current_vms = []
        self.workload_history = []
        
    def monitor(self, current_workload, vm_utilization):
        """Monitor phase: Collect system metrics"""
        metrics = {
            'workload': current_workload,
            'vm_utilization': vm_utilization,
            'vm_count': len(self.current_vms)
        }
        self.workload_history.append(metrics)
        return metrics
    
    def analyze(self, metrics):
        """Analyze phase: Process collected metrics and predict future workload"""
        if len(self.workload_history) < 2:
            return metrics['workload']
            
        # Linear regression for workload prediction
        X = np.array(range(len(self.workload_history))).reshape(-1, 1)
        y = np.array([m['workload'] for m in self.workload_history])
        
        model = LinearRegression()
        model.fit(X, y)
        
        next_timestamp = len(self.workload_history)
        predicted_workload = model.predict([[next_timestamp]])[0]
        
        return predicted_workload
    
    def determine_state(self, utilization):
        """Determine current system state based on utilization"""
        if utilization < self.lower_threshold:
            return 'underutilized'
        elif utilization > self.upper_threshold:
            return 'overutilized'
        return 'normal'
    
    def get_reward(self, state, action, new_utilization):
        """Calculate reward for state-action pair"""
        if state == 'normal' and new_utilization >= self.lower_threshold and new_utilization <= self.upper_threshold:
            return 1.0
        elif state == 'underutilized' and action == 'scale_down':
            return 0.5
        elif state == 'overutilized' and action == 'scale_up':
            return 0.5
        return -0.1
    
    def plan(self, current_state, predicted_workload):
        """Planning phase: Use Q-learning to decide action"""
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(self.actions)
        
        # Exploitation
        return max(self.q_table[current_state].items(), key=lambda x: x[1])[0]
    
    def execute(self, action, current_vms):
        """Execute phase: Implement the decided action"""
        if action == 'scale_up':
            # Add new VM
            new_vm = {'type': 'small', 'utilization': 0.0}
            current_vms.append(new_vm)
        elif action == 'scale_down' and current_vms:
            # Remove least utilized VM
            current_vms.pop()
        
        return current_vms
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value for state-action pair"""
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[state][action] = new_value
    
    def run_mape_loop(self, current_workload, current_utilization):
        """Run complete MAPE-K loop"""
        # Monitor
        metrics = self.monitor(current_workload, current_utilization)
        
        # Analyze
        predicted_workload = self.analyze(metrics)
        
        # Determine current state
        current_state = self.determine_state(current_utilization)
        
        # Plan
        action = self.plan(current_state, predicted_workload)
        
        # Execute
        new_vms = self.execute(action, self.current_vms)
        
        # Calculate new utilization and reward
        new_utilization = current_workload / max(len(new_vms), 1)
        reward = self.get_reward(current_state, action, new_utilization)
        
        # Update Q-values
        new_state = self.determine_state(new_utilization)
        self.update_q_value(current_state, action, reward, new_state)
        
        self.current_vms = new_vms
        
        return {
            'action': action,
            'vm_count': len(new_vms),
            'utilization': new_utilization,
            'predicted_workload': predicted_workload,
            'reward': reward
        }