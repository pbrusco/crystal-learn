# http://www.saedsayad.com/decision_tree.htm

abstract class Tree
end

class Node < Tree
  property :left_child
  property :right_child
  property :feature_index
  property :split_value

  @left_child : Tree
  @right_child : Tree
  @split_value : String | Float32 | Float64 #TODO: Change for generic types when crystal is ready :)

  def initialize(@feature_index : Int32, @split_value)
    @right_child = EmptyTree.new
    @left_child = EmptyTree.new
  end

  def depth
    Math.max(@left_child.depth, @right_child.depth) + 1
  end

  def children
    [@left_child, @right_child]
  end

  def show(column_names, level)
    tabs = "\t" * level
    feature_name = column_names ? column_names[@feature_index] : "F#{@feature_index}"

    puts tabs + "Node(feature=#{feature_name})"
    children.each do |c|
      puts tabs + "split_by: #{@split_value}"
      c.show(column_names, level+1)
    end
  end
end

class Leaf < Tree
  property :tags

  def initialize(@tags : Array(String) | Array(Float32) | Array(Float64))  # TODO: Change for generic type when crystal is ready :)
  end

  def depth
    0
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tags.mode})"
  end
end

class EmptyTree < Tree
  def show(column_names, level)
  end

  def depth
    0
  end
end
