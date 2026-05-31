import json
from confluent_kafka import Consumer, Producer

KAFKA_BROKER = 'localhost:9092' 
CONSUMER_GROUP = 'ashpaz_validator_group'
TOPIC_IN = 'ashpaz.order'
TOPIC_VALID = 'ashpaz.valid'
TOPIC_ERROR = 'ashpaz.error_log'

consumer = Consumer({
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': CONSUMER_GROUP,
    'auto.offset.reset': 'earliest'
})

producer = Producer({'bootstrap.servers': KAFKA_BROKER})
consumer.subscribe([TOPIC_IN])

print(f"🎧 Listening for orders on '{TOPIC_IN}'...")

try:
    while True:
        msg = consumer.poll(1.0)
        
        if msg is None:
            continue
        if msg.error():
            print(f"⚠️ Consumer error: {msg.error()}")
            continue

        raw_data = msg.value().decode('utf-8')
        try:
            order_data = json.loads(raw_data)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON. Skipping message.")
            continue

        errors = []
        error_types = []

        phone = order_data.get("phone_number", "")
        if not (phone.startswith("+91") or phone.startswith("080")):
            error_types.append("INVALID_PHONE")
            errors.append(f"Phone number '{phone}' is invalid.")

        req_online = order_data.get("request_online", False)
        req_table = order_data.get("request_table", False)
        if req_online and req_table:
            error_types.append("ORDER_MODE_CONFLICT")
            errors.append("Order cannot be both online and a table request.")

        items = order_data.get("items", [])
        calculated_price = sum(item.get("unit_price", 0) * item.get("quantity", 0) for item in items)
        order_price = order_data.get("order_price", 0)
        
        if calculated_price != order_price:
            error_types.append("PRICE_MISMATCH")
            errors.append(f"Calculated price ({calculated_price}) != order_price ({order_price}).")

        if errors:
            error_payload = {
                "order_id": order_data.get("order_id", "UNKNOWN"),
                "error_type": "MULTI" if len(error_types) > 1 else error_types[0],
                "error_reason": errors,
                "original_order": order_data
            }
            producer.produce(TOPIC_ERROR, value=json.dumps(error_payload).encode('utf-8'))
            print(f"❌ INVALID ORDER [{error_payload['order_id']}]: Routed to {TOPIC_ERROR}")
            
            print(json.dumps(error_payload, indent=2))
        
        else:
            producer.produce(TOPIC_VALID, value=raw_data.encode('utf-8'))
            print(f"✅ VALID ORDER [{order_data.get('order_id', 'UNKNOWN')}]: Routed to {TOPIC_VALID}")

        producer.poll(0)

except KeyboardInterrupt:
    print("\n🛑 Process interrupted by user. Shutting down...")
finally:
    consumer.close()
    producer.flush()