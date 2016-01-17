package annotation;

import org.grouplens.grapht.annotation.DefaultDouble;
import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;

@Documented
@DefaultDouble(0.5)
@Parameter(Double.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface Alpha {
}
