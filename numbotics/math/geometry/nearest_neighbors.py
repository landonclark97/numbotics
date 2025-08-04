import numpy as np
import faiss



class ApproximateNearestNeighborsIndex:

    def __init__(self, dim: int):
        self._dim = dim
        self._index = faiss.IndexFlatL2(dim)
        self._points = {}
        self._next_id = 0


    def add_points(self, points: np.ndarray):
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != self._dim:
            raise ValueError(f"Point dimension {points.shape[1]} does not match index dimension {self._dim}")

        ids = []
        for point in points:
            point_id = self._next_id
            self._points[point_id] = point
            ids.append(point_id)
            self._next_id += 1

        self._index.add(points.astype(np.float32))
        return ids


    def add_point(self, point: np.ndarray):
        if point.ndim != 1:
            raise ValueError(f"Point should be a 1D array, got shape {point.shape}")
        return self.add_points(point.reshape(1, -1))[0]


    def remove_points(self, ids: list[int]):
        for point_id in ids:
            if point_id in self._points:
                del self._points[point_id]
        self._rebuild_index()


    def remove_point(self, point_id: int):
        self.remove_points([point_id])


    def _rebuild_index(self):
        self._index.reset()
        if self._points:
            all_points = np.vstack(list(self._points.values())).astype(np.float32)
            self._index.add(all_points)


    def query(self, query: np.ndarray, k: int = 1, return_labels: bool = False, flatten: bool = True):
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self._dim:
            raise ValueError(f"Query dimension {query.shape[1]} does not match index dimension {self._dim}")

        _, indices = self._index.search(query.astype(np.float32), k=min(k, len(self._points)))

        items = np.array([self._points[idx] for idx in indices[0]])
        labels = indices[0]

        if flatten and k == 1:
            items = items[0]
            labels = labels[0]

        if return_labels:
            return items, labels
        return items


    def nearest(self, query: np.ndarray, return_labels: bool = False):
        return self.query(query, k=1, return_labels=return_labels, flatten=True)


    def k_nearest(self, query: np.ndarray, k: int, return_labels: bool = False):
        return self.query(query, k=k, return_labels=return_labels)


    def __len__(self):
        return len(self._points)
