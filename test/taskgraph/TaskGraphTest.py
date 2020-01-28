from taskgraph import TaskGraph
from taskgraph import Task

_task_graph_cache_dir = "/tmp/taskgraph/"
_workers = 1
taskgraph = TaskGraph(taskgraph_cache_dir_path=_task_graph_cache_dir, n_workers=_workers)

#task1 = Task(task_name="task1")


def task1():
    print("Task 1")

def task2():
    print("Task 2")



taskgraph.add_task(func=task2)
taskgraph.add_task(func=task1)

taskgraph.close()
taskgraph.join()