/**
 * S-hook with different sized ends.
 */

// Large end inner diameter
//diameter_large = 19.4;

diameter_large = 26.2;

// Small end inner diameter.
//diameter_small = 19.4;

diameter_small = 26.2;

// Hook body diameter.
//diameter_body = 4;
diameter_body = 5.8;

hook_angle = 90;

module s_hook(diameter_large=41, diameter_small=5, diameter_body=5, hook_angle=90) {
	$fa = 0.1;
	$fs = 0.25;

	radius=(diameter_small / 2) + (diameter_body / 2);
    angle=hook_angle;
	translate([radius, 0, 0]) rotate([0, 0, 180]) difference() {
			rotate_extrude() translate([radius, 0, 0]) circle(r = diameter_body / 2); 
			translate([0, 0, -diameter_body / 2]) linear_extrude(diameter_body) pie_slice(r=(2 * radius) + (2 * diameter_body), a=angle);
	}

	radius=(diameter_large / 2) + (diameter_body / 2);
    angle=hook_angle;
	translate([-radius, 0, 0]) rotate([0, 0, 0]) difference() {
			rotate_extrude() translate([radius, 0, 0]) circle(r = diameter_body / 2); 
			translate([0, 0, -diameter_body / 2]) linear_extrude(diameter_body) pie_slice(r=(2 * radius) + (2 * diameter_body), a=angle);
	}
}

module pie_slice(r=3.0, a=30) {
	intersection() {
		circle(r=r);
		square(r);
		rotate(a-90) square(r);
	}
}

rotate([0,90,0])
s_hook(diameter_large=diameter_large, diameter_small=diameter_small, diameter_body=diameter_body, hook_angle=hook_angle);