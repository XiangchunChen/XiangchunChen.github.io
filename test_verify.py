import os

def test_pages():
    project_dir = r"e:\ClaudeCodeFiles\XiangchunChen.github.io"
    
    demos_path = os.path.join(project_dir, "demos.html")
    demo2_path = os.path.join(project_dir, "demo2", "demo2.html")
    index_path = os.path.join(project_dir, "index.html")
    
    assert os.path.exists(demos_path), "demos.html fails to exist"
    assert os.path.exists(demo2_path), "demo2.html fails to exist"
    
    with open(demos_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "href=\"demo/demo.html\"" in content, "demos.html should link to demo 1"
        assert "href=\"demo2/demo2.html\"" in content, "demos.html should link to demo 2"
        
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "href=\"demos.html\"" in content, "index.html should link to demos.html"
        
    print("All tests passed successfully.")

if __name__ == "__main__":
    test_pages()
