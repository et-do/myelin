"""Tests for thread-safety of core data stores."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from myelin.config import MyelinSettings
from myelin.recall.activation import HebbianTracker
from myelin.store.hippocampus import Hippocampus
from myelin.store.neocortex import SemanticNetwork


def _make_network() -> SemanticNetwork:
    return SemanticNetwork(db_path=Path(":memory:"))


class TestSemanticNetworkConcurrency:
    def test_concurrent_add_entity(self) -> None:
        net = _make_network()
        errors: list[Exception] = []

        def add_entities(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    net.add_entity(f"entity_{i}", entity_type="test")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_entities, args=(i * 50,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert net.entity_count() == 200

    def test_concurrent_add_relationship(self) -> None:
        net = _make_network()
        for i in range(20):
            net.add_entity(f"e{i}")
        errors: list[Exception] = []

        def add_rels(offset: int) -> None:
            try:
                for i in range(10):
                    net.add_relationship(
                        f"e{offset}", "relates_to", f"e{(offset + i + 1) % 20}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_rels, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert net.relationship_count() > 0

    def test_concurrent_read_write(self) -> None:
        net = _make_network()
        net.add_entity("seed", entity_type="test")
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    net.add_entity(f"w_{i}")
                    net.add_relationship("seed", "links", f"w_{i}")
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(50):
                    net.get_entity("seed")
                    net.get_relationships("seed")
                    net.entity_count()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestHebbianTrackerConcurrency:
    def test_concurrent_reinforce(self, tmp_path: Path) -> None:
        tracker = HebbianTracker(db_path=tmp_path / "hebb.db")
        errors: list[Exception] = []

        def reinforce_batch(ids: list[str]) -> None:
            try:
                for _ in range(20):
                    tracker.reinforce(ids)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reinforce_batch, args=(["a", "b", "c"],)),
            threading.Thread(target=reinforce_batch, args=(["b", "c", "d"],)),
            threading.Thread(target=reinforce_batch, args=(["a", "c", "d"],)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        tracker.close()

        assert not errors

    def test_concurrent_reinforce_and_cleanup(self, tmp_path: Path) -> None:
        tracker = HebbianTracker(db_path=tmp_path / "hebb.db")
        tracker.reinforce(["x", "y", "z"])
        errors: list[Exception] = []

        def reinforce_loop() -> None:
            try:
                for _ in range(30):
                    tracker.reinforce(["x", "y"])
            except Exception as e:
                errors.append(e)

        def cleanup_loop() -> None:
            try:
                for _ in range(30):
                    tracker.cleanup({"x", "y", "z"})
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reinforce_loop),
            threading.Thread(target=cleanup_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        tracker.close()

        assert not errors


class TestHippocampusConcurrency:
    @pytest.fixture()
    def hipp(self, tmp_path: Path) -> Hippocampus:
        return Hippocampus(cfg=MyelinSettings(data_dir=tmp_path / ".myelin"))

    def test_concurrent_store(self, hipp: Hippocampus) -> None:
        errors: list[Exception] = []
        topics = [
            "Python decorators and metaclasses",
            "Kubernetes pod networking",
            "PostgreSQL query optimization",
        ]

        def store_batch(prefix: str, topic: str) -> None:
            try:
                for i in range(3):
                    hipp.store(
                        f"{prefix}: {topic} lesson {i} with unique detail {prefix}_{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(
                target=store_batch,
                args=(f"thread_{t}", topics[t]),
            )
            for t in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Some may be dedup'd; just verify no crashes and data stored
        assert hipp.count() >= 3

    def test_concurrent_store_and_recall(self, hipp: Hippocampus) -> None:
        # Pre-populate
        for i in range(5):
            hipp.store(f"Python is a great programming language, fact {i}")
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(5):
                    hipp.store(f"JavaScript is also useful for web dev, fact {i}")
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(5):
                    hipp.recall("programming language")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_store_and_count(self, hipp: Hippocampus) -> None:
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(10):
                    hipp.store(f"Concurrent count test memory number {i}")
            except Exception as e:
                errors.append(e)

        def counter() -> None:
            try:
                for _ in range(10):
                    hipp.count()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=counter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
