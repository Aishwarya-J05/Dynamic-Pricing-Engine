from __future__ import annotations

import json
from typing import Any

from kafka import KafkaConsumer, KafkaProducer

from app.config import Settings


class PricingEventProducer:
    def __init__(self, settings: Settings):
        self.producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        self.order_topic = settings.kafka_topic_orders
        self.click_topic = settings.kafka_topic_clicks

    def publish_order(self, payload: dict[str, Any]) -> None:
        self.producer.send(self.order_topic, payload)
        self.producer.flush()

    def publish_click(self, payload: dict[str, Any]) -> None:
        self.producer.send(self.click_topic, payload)
        self.producer.flush()


class PricingEventConsumer:
    def __init__(self, settings: Settings, topic: str):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="dynamic-pricing-engine",
        )

    def poll_forever(self):
        for message in self.consumer:
            yield message.value
