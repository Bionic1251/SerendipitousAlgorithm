package annotation;

import org.grouplens.grapht.annotation.DefaultBoolean;
import org.grouplens.lenskit.core.Parameter;

import javax.inject.Qualifier;
import java.lang.annotation.*;

@Documented
@DefaultBoolean(false)
@Parameter(Boolean.class)
@Qualifier
@Target({ElementType.METHOD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
public @interface TFIDF {
}
