import os

def test_verify_html():
    file_path = "e:/ClaudeCodeFiles/XiangchunChen.github.io/demo2/demo2.html"
    assert os.path.exists(file_path), "demo2.html does not exist!"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for background color matching the cream tone
    assert "#f3ede1" in content, "Background color #f3ede1 is missing."

    # Check for proper title text
    assert "PolyClaw" in content, "Main 'PolyClaw' text is missing."
    assert "Key Features" in content, "'Key Features' section header is missing."

    # Check for the correct video file
    assert "EdgeClaw%20Demo.mp4" in content, "The demo video is missing."

    # Verify key feature texts
    assert "Local-First Architecture" in content, "Feature text missing."
    assert "Privacy-by-Design" in content, "Feature text missing."
    assert "OpenClaw Compatible" in content, "Feature text missing."
    assert "Edge-Optimized" in content, "Feature text missing."
    assert "Full User Control" in content, "Feature text missing."

    print("All verification steps passed successfully!")

if __name__ == "__main__":
    test_verify_html()
