import heapq

"""
dijkstra算法：确定起点，找到每个节点距离起点的最短路径
limitations: 加权路径为正，固定起点
"""
def dijkstra(graph, start):
    # 初始化距离表，所有顶点的距离为无穷大
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0  # 起始点到自身的距离为0

    # 优先队列（最小堆）存储 (距离, 顶点)
    priority_queue = [(0, start)]

    while priority_queue:
        # 取出距离起始点最近的顶点
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果当前顶点的距离已经不是最小值，跳过它
        if current_distance > distances[current_vertex]:
            continue

        # 遍历邻接顶点并更新距离
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight

            # 如果找到更短的路径，更新距离并将邻接顶点加入队列
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances
