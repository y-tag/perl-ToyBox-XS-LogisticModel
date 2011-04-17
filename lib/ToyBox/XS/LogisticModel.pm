package ToyBox::XS::LogisticModel;

use 5.0080;
use strict;
use warnings;

require Exporter;

our $VERSION = '0.01';

require XSLoader;
XSLoader::load('ToyBox::XS::LogisticModel', $VERSION);

sub add_instance {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    die "No params: label" unless defined($params{label});
    my $attributes = $params{attributes};
    my $label      = $params{label};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;

    foreach my $l (@$label) {
        $self->xs_add_instance(\%copy_attr, $l);
    }
    1;
}

sub train{
    my ($self, %params) = @_;

    die "max_iterations should be greater than 0" if defined($params{max_iterations}) && $params{max_iterations} <= 0;
    die "max_linesearch should be greater than 0" if defined($params{max_linesearch}) && $params{max_linesearch} <= 0;

    my $max_iterations = $params{max_iterations} || 100;
    my $max_linesearch = $params{max_linesearch} || 20;

    $self->xs_train($max_iterations, $max_linesearch);
    1;
}

sub predict {
    my ($self, %params) = @_;

    die "No params: attributes" unless defined($params{attributes});
    my $attributes = $params{attributes};
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $result = $self->xs_predict($attributes);

    $result;
}


1;
__END__
=head1 NAME

ToyBox::XS::LogisticModel - Discriminant Analysis with Logistic Model

=head1 SYNOPSIS

  use ToyBox::XS::LogisticModel;

  my $lm = ToyBox::XS::LogisticModel->new();
  
  $lm->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $lm->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $lm->train(max_iterations => 100, max_linesearch => 20);
  
  my $probs = $lm->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

This module implements a logistic model.

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

Copyright (C) 1990, Jorge Nocedal

Copyright (C) 2007, Naoaki Okazaki

Copyright (C) 2011, TAGAMI Yukihiro

This software is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>

=head1 REFERENCE

=over

=item
J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage (1980)
, Mathematics of Computation 35, pp. 773-782.

=item
D.C. Liu and J. Nocedal. On the Limited Memory Method for Large Scale
Optimization (1989), Mathematical Programming B, 45, 3, pp. 503-528.

=item
Jorge Nocedal's Fortran 77 implementation,
L<http://www.ece.northwestern.edu/~nocedal/lbfgs.html>

=item
Naoaki Okazaki's C implementation (liblbfgs),
L<http://www.chokkan.org/software/liblbfgs/index.html>

=back

