package annotation;

import org.grouplens.grapht.annotation.DefaultDouble;
import org.grouplens.grapht.annotation.DefaultInteger;
import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;

@Documented
@DefaultDouble(3.0)
@Parameter(Double.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface Threshold {
}
