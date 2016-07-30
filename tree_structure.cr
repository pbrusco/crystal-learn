# http://www.saedsayad.com/decision_tree.htm

abstract struct Tree
end

struct Node(T1) < Tree
  property :children
  property :feature_index

  @children : Array(Tree)

  def initialize(@feature_index : Int32, @values : Array(T1))
    @children = Array(Tree).new(@values.size)
  end

  def children_with_value(value)
    index = @values.index(value)
    if index
      @children[index]
    else
      raise "Value: #{value} not found in the possible values of the feature: F#{@feature_index} #{@values}"
    end

  end

  def show(column_names, level)
    tabs = "\t" * level
    feature_name = column_names ? column_names[@feature_index] : "F#{@feature_index}"

    puts tabs + "Node(feature=#{feature_name})"
    @children.zip(@values).each do |c, v|
      puts tabs + "value: #{v}"
      c.show(column_names, level+1)
    end
  end

end

struct Leaf(T2) < Tree
  property :tags

  def initialize(@tags : Array(T2))
  end

  def show(column_names, level)
    tabs = "\t" * level
    class_name = column_names ? column_names.last : "class: "
    puts tabs + "Hoja(#{class_name}: #{@tags})"
  end
end

struct EmptyTree < Tree
  def show(column_names, level)
  end
end
