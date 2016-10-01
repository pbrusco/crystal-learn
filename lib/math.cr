module Math
  def Math.sin(vec : Array(T))
    vec.map { |x| Math.sin(x) }
  end
end
