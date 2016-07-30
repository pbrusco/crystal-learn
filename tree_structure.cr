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
  @split_value : String

  def initialize(@feature_index : Int32, @split_value : String)
    @right_child = EmptyTree.new
    @left_child = EmptyTree.new
  end

  def children
    [@left_child, @right_child]
  end

  def show(column_names, level)
    tabs = "\t" * level
    feature_name = column_names ? column_names[@feature_index] : "F#{@feature_index}"

    puts tabs + "Node(feature=#{feature_name})"
    children.zip(@values).each do |c, v|
      puts tabs + "value: #{v}"
      c.show(column_names, level+1)
    end
  end
end

class Leaf < Tree
  property :tags

  def initialize(@tags : Array(String) | Array(Float32))  # TODO: Change for generic type when crystal is ready :)
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tags})"
  end
end

class EmptyTree < Tree
  def show(column_names, level)
  end
end
