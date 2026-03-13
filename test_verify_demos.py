import os

def test_verify_demos_html():
    file_path = "e:/ClaudeCodeFiles/XiangchunChen.github.io/demos.html"
    assert os.path.exists(file_path), "demos.html does not exist!"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for background color matching the cream tone
    assert "#f3ede1" in content, "Background color #f3ede1 is missing."

    # Verify content preservation
    assert "PolyScheduler" in content, "PolyScheduler text missing."
    assert "PolyClaw" in content, "PolyClaw text missing."
    assert "Research Demonstrations" in content, "Header missing."
    assert "demo/demo.html" in content, "Link to demo1 missing."
    assert "demo2/demo2.html" in content, "Link to demo2 missing."

    # Look for light theme specific classes or styles
    assert "bg-white" in content or "background-color: #ffffff" in content, "Missing white card background style."

    print("demos.html verification passed successfully!")

if __name__ == "__main__":
    test_verify_demos_html()
