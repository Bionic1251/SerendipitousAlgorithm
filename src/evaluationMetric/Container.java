package evaluationMetric;

public class Container<T extends Comparable<T>> implements Comparable<Container> {
	private Long id;
	private T value;

	public Container(Long id, T value) {
		this.id = id;
		this.value = value;
	}

	public Long getId() {
		return id;
	}

	public T getValue() {
		return value;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public void setValue(T value) {
		this.value = value;
	}

	@Override
	public int compareTo(Container o) {
		Container<T> obj = (Container<T>) o;
		return value.compareTo(obj.getValue());
	}

	@Override
	public boolean equals(Object obj) {
		Container container = (Container) obj;
		return id.equals(container.getId());
	}
}
