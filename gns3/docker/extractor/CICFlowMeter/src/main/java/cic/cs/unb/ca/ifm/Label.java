package cic.cs.unb.ca.ifm;

import java.util.Objects;

public class Label {
    public final static Label NO_LABEL = new Label("No Label");
    public final static Label BENIGN = new Label("benign");

    private final String label;

    public static Label[] getLabelsFromAttacksString(final String attacksString) {
        final String[] splitString = attacksString.split(":");
        final Label[] labels = new Label[splitString.length + 1];
        labels[0] = BENIGN;
        int i = 0;
        for(final String label : splitString) {
            labels[++i] = new Label(label);
        }
        return labels;
    }

    public Label(final String label) {
        if(label == null) {
            throw new NullPointerException("Label must not be null");
        }
        this.label = label;
    }

    @Override
    public String toString() {
        return label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o instanceof Label) {
            Label otherLabel = (Label) o;
            return label.equals(otherLabel.label);
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return label.hashCode();
    }
}
