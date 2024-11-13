[PYTHON]
import gc
from typing import List

class Node:
    def __init__(self, node):
        self.node = node

    def garbage_collect(self) -> None:
        gc.collect()
        print("Garbage collected")

    def get_children(self) -> List[Node]:
        children = []
        for child in self.node.children:
            children.append(Node(child))
        return children
[/PYTHON]
[PHP]
class Node
{
    private $node;

    public function __construct($node)
    {
        $this->node = $node;
    }

    public function garbageCollect()
    {
        gc_collect_cycles();
        echo "Garbage collected";
    }

    public function getChildren()
    {
        $children = array();
        foreach ($this->node->children as $child) {
            $children[] = new Node($child);
        }
        return $children;
    }
}
[/PHP]
[JAVA]
import java.util.ArrayList;
import java.util.List;

public class Node {
    private final Object node;

    public Node(Object node) {
        this.node = node;
    }

    public void garbageCollect() {
        System.gc();
        System.out.println("Garbage collected");
    }

    public List<Node> getChildren() {
        List<Node> children = new ArrayList<>();
        for (Object child : this.node.getChildren()) {
            children.add(new Node(child));
        }
        return children;
    }
}
[/JAVA]
[C#]
using System;
using System.Collections.Generic;

public class Node
{
    private readonly object node;

    public Node(object node)
    {
        this.node = node;
    }

    public void GarbageCollect()
    {
        GC.Collect();
        Console.WriteLine("Garbage collected");
    }

    public List<Node> GetChildren()
    {
        var children = new List<Node>();
        foreach (var child in this.node.GetChildren())
        {
            children.Add(new Node(child));
        }
        return children;
    }
}
[/C#]
[JS]
class Node {
  constructor(node) {
    this.node = node;
  }

  garbageCollect() {
    gc();
    console.log("Garbage collected");
  }

  getChildren() {
    const children = [];
    for (const child of this.node.children) {
      children.push(new Node(child));
    }
    return children;
  }
}
[/JS]
[TYPESCRIPT]
class Node {
    constructor(private node: any) {}

    garbageCollect() {
        gc();
        console.log("Garbage collected");
    }

    getChildren(): Node[] {
        const children = [];
        for (const child of this.node.children) {
            children.push(new Node(child));
        }
        return children;
    }
}
[/TYPESCRIPT]
[SCALA]
class Node(val node: Any) {
  def garbageCollect(): Unit = {
    gc()
    println("Garbage collected")
  }

  def getChildren(): List[Node] = {
    val children = new java.util.ArrayList[Node]
    for (child <- this.node.children) {
      children.add(new Node(child))
    }
    return children
  }
}
[/SCALA]
