from graphviz import Digraph
import os

def generate_agent_hierarchy():
    dot = Digraph(comment='Agent Hierarchy', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.8')
    
    # Global node styles
    dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='12', margin='0.2,0.1')
    dot.attr('edge', fontname='Arial', fontsize='10')

    # User Node
    dot.node('User', 'User Request', fillcolor='#E1F5FE', color='#0277BD')

    # Coordinator Cluster
    with dot.subgraph(name='cluster_Coordinator') as c:
        c.attr(label='Coordinator Agent (SequentialAgent)', style='dashed', color='#546E7A', fontcolor='#546E7A', bgcolor='#FAFAFA')
        
        # Step 1: Parallel Agent
        with c.subgraph(name='cluster_Parallel') as p:
            p.attr(label='Step 1: Price Search (ParallelAgent)', style='filled', color='#FFF3E0', bgcolor='#FFF3E0')
            
            # Vision Agents inside Parallel Agent
            p.node('Tesco', 'VisionAgent\n(Supermarket 3)', fillcolor='#FFFFFF', color='#FF9800')
            p.node('Morrisons', 'VisionAgent\n(Supermarket 2)', fillcolor='#FFFFFF', color='#FF9800')
            p.node('Sainsburys', 'VisionAgent\n(Supermarket 1)', fillcolor='#FFFFFF', color='#FF9800')
            
        # Result Accumulator (Shared State)
        # We place it "between" the steps conceptually in the diagram
        c.node('Accumulator', 'Result Accumulator\n(Shared List)', shape='cylinder', fillcolor='#E0F2F1', color='#00695C')
        
        # Step 2: Optimization Agent
        with c.subgraph(name='cluster_Optimizer') as o:
            o.attr(label='Step 2: OptimizationAgent', style='filled', color='#E8F5E9', bgcolor='#E8F5E9')
            o.node('Optimizer', 'OptimizationAgent', fillcolor='#FFFFFF', color='#4CAF50')

        # Edges inside Coordinator
        # Parallel Agent -> Vision Agents (Implicit in containment, but we show data flow)
        
        # Vision Agents write to Accumulator
        c.edge('Tesco', 'Accumulator')
        c.edge('Morrisons', 'Accumulator')
        c.edge('Sainsburys', 'Accumulator')
        
        # Accumulator feeds Optimizer
        c.edge('Accumulator', 'Optimizer')

    # Final Output
    dot.node('Report', 'Savings Report', shape='note', fillcolor='#F3E5F5', color='#7B1FA2')

    # Main Flow Edges
    dot.edge('User', 'Tesco', lhead='cluster_Coordinator') 
    dot.edge('Optimizer', 'Report')

    # Render
    output_path = 'docs/images/agent_hierarchy'
    dot.render(output_path, cleanup=True)
    print(f"Diagram generated at {output_path}.png")

if __name__ == "__main__":
    generate_agent_hierarchy()
