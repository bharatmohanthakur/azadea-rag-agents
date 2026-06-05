
import asyncio
import json
import re
import sys

# Mocking the generator logic from rag_server_ls.py to verify the fix
# We will duplicate the logic here to test it in isolation first

async def simulate_streaming(text, request_id, use_fixed_logic=False):
    print(f"--- Simulating Streaming (Fixed Logic: {use_fixed_logic}) ---")
    
    if not use_fixed_logic:
        # ORIGINAL LOGIC
        # If no code blocks, use simple word-by-word streaming
        words = text.split()
        for i, word in enumerate(words):
            text_chunk = word if i == 0 else f" {word}"
            yield f"data: {json.dumps({'type': 'token', 'text': text_chunk}, ensure_ascii=False)}\n\n"
    else:
        # NEW LOGIC
        # import re (already imported at top)
        
        # Detect code blocks (simplified for this test)
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        # For this test, we assume no code blocks to focus on text splitting
        
        # Re-implementing the propose fix logic
        # Split by whitespace but keep the delimiters (capture group)
        # This will separate words and whitespace/newlines
        tokens = re.split(r'(\s+)', text)
        
        for token in tokens:
            if not token: 
                continue
                
            # If it's whitespace (including newlines), yield it directly
            if re.match(r'^\s+$', token):
                yield f"data: {json.dumps({'type': 'token', 'text': token, 'request_id': request_id}, ensure_ascii=False)}\n\n"
            else:
                # It's a word
                yield f"data: {json.dumps({'type': 'token', 'text': token, 'request_id': request_id}, ensure_ascii=False)}\n\n"

async def main():
    test_text = "Line 1.\nLine 2 is here."
    request_id = "req-123"
    
    print(f"Original Text:\n{repr(test_text)}\n")
    
    # Test Original Logic
    reassembled_original = ""
    print("Output from Original Logic:")
    async for chunk in simulate_streaming(test_text, request_id, use_fixed_logic=False):
        # Parse data
        line = chunk.strip()
        if line.startswith("data: "):
            data = json.loads(line[6:])
            text = data.get('text', '')
            reassembled_original += text
            # print(f"Chunk: {repr(text)}")
    
    print(f"Reassembled Original:\n{repr(reassembled_original)}")
    if reassembled_original != test_text:
        print("❌ FAIL: Original logic does not preserve formatting (expected).")
    else:
        print("✅ PASS: Original logic somehow preserved formatting (unexpected).")

    print("\n" + "="*30 + "\n")

    # Test Fixed Logic
    reassembled_fixed = ""
    chunk_has_request_id = True
    print("Output from New Logic:")
    async for chunk in simulate_streaming(test_text, request_id, use_fixed_logic=True):
        line = chunk.strip()
        if line.startswith("data: "):
            data = json.loads(line[6:])
            text = data.get('text', '')
            if 'request_id' not in data or data['request_id'] != request_id:
                chunk_has_request_id = False
            reassembled_fixed += text
            # print(f"Chunk: {repr(text)}")

    print(f"Reassembled Fixed:\n{repr(reassembled_fixed)}")
    
    success = True
    if reassembled_fixed != test_text:
        print("❌ FAIL: Fixed logic does NOT match original text.")
        success = False
    else:
        print("✅ PASS: Fixed logic matches original text.")

    if not chunk_has_request_id:
        print("❌ FAIL: chunks missing request_id")
        success = False
    else:
        print("✅ PASS: chunks contain request_id")

    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
