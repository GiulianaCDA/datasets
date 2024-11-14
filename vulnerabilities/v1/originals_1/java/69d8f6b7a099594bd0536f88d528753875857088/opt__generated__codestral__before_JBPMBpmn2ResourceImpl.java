import bpsim.impl.BpsimPackageImpl;
import org.eclipse.bpmn2.*;
import org.eclipse.bpmn2.util.Bpmn2ResourceImpl;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.xmi.XMLLoad;
import org.eclipse.emf.ecore.xmi.XMLResource;
import org.eclipse.emf.ecore.xmi.XMLSave;
import org.eclipse.emf.ecore.xmi.impl.XMLLoadImpl;
import org.eclipse.emf.ecore.resource.Resource;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

public class JBPMBpmn2ResourceImpl extends Bpmn2ResourceImpl {

    public JBPMBpmn2ResourceImpl(URI uri) {
        super(uri);
        this.getDefaultLoadOptions().put(XMLResource.OPTION_DEFER_IDREF_RESOLUTION, true);
        this.getDefaultLoadOptions().put(XMLResource.OPTION_DISABLE_NOTIFY, true);
        // Switch off DTD external entity processing
        Map<String, Boolean> parserFeatures = new HashMap<>();
        parserFeatures.put("http://xml.org/sax/features/external-general-entities", false);
        this.getDefaultLoadOptions().put(XMLResource.OPTION_PARSER_FEATURES, parserFeatures);
        this.getDefaultSaveOptions().put(XMLResource.OPTION_ENCODING, "UTF-8");
        this.getDefaultSaveOptions().put(XMLResource.OPTION_PROCESS_DANGLING_HREF, XMLResource.OPTION_PROCESS_DANGLING_HREF_DISCARD);
    }

    @Override
    protected XMLSave createXMLSave() {
        prepareSave();
        return new JBPMXMLSaveImpl(createXMLHelper()) {
            @Override
            protected boolean shouldSaveFeature(EObject o, EStructuralFeature f) {
                if (Bpmn2Package.eINSTANCE.getDocumentation_Text().equals(f))
                    return false;
                if (Bpmn2Package.eINSTANCE.getFormalExpression_Body().equals(f))
                    return false;
                return super.shouldSaveFeature(o, f);
            }
        };
    }

    @Override
    protected XMLLoad createXMLLoad() {
        DroolsPackageImpl.init();
        BpsimPackageImpl.init();
        return new XMLLoadImpl(this) {
            @Override
            public void load(InputStream inputStream, Map<?, ?> options) throws IOException {
                // Add input validation and sanitization here if needed
                super.load(inputStream, options);
            }

            @Override
            protected void handleException(Exception exception) {
                // Handle exceptions properly without exposing sensitive information to the client
                // For example, log the error and throw a generic exception
                exception.printStackTrace();
                throw new RuntimeException("An error occurred while loading the resource");
            }
        };
    }
}
