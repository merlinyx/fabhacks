// end-to-end length
length = 425;//435; 
// approximate height to top of diagonal, from center of bar to center of bar
height = 110;//115; 
// bar diameter
bar_d = 15;//6.35;
// corner bend diameter
bend_d = 20; //25;
// hook diameter
hook_d = 63;//40;
// how long is the neck? measured from top of diagonal to top of hook
hook_h = 100;//105;

module hanger(length, height, bar_d, bend_d, hook_d, hook_h) {
/* [Advanced] */
// get a smoother profile by setting to the highest value your printer can manage. 45 degrees works for most printers.
overhang_angle = 45; // [0:90]
// space between pins on pin plate
pin_buffer = 5;
// space between mating parts
tolerance = 0.2; 

// facets
$fn=32;
/* [Hidden] */
// bar radius
bar_r = bar_d / 2;
// bend radius
bend_r = bend_d / 2;
// compensate for truncation of radius due to low poly count. I.e. circumscribe the circle with the polygon, not the other way around
//thickness = bar_d*cos(180/(bar_facets)); 
thickness = bar_d;
// pin diameter
pin_d = bar_d;
// left-to-right plate height
plate_h = pin_d * 2 + pin_buffer + 4;
// top to bottom plate width
plate_w = pin_d * 2;
// thickness of bottom plate
plate_t = thickness / 2;
// left-to-right crossbeam height
crossbeam_h = pin_d;
// length of horizontal bar
bottom_l = length /2 - bend_r - plate_h / 2;
// corner angle
echo(plate_h);
echo(plate_w);
echo(crossbeam_h);
echo(bottom_l);
//corner_a = atan((height - bend_d)/ bottom_l);
corner_a = atan((height - bend_d)/ bottom_l);
echo(corner_a);
// length of diagonal bar
top_l = sqrt((bottom_l + tan(corner_a)*bend_r) * (bottom_l + tan(corner_a)*bend_r) + (height - bend_d)*(height - bend_d));
echo(top_l);
bend_a = 180 - corner_a;
// hook radius
hook_r = hook_d / 2;
// distance to move the horizontal bar
bottom_offset = -sin(corner_a)*top_l - bend_d - bar_r;
echo(bottom_offset);
// extrude a ring of the given profile
// outer - radius of the profile
// inner - radius of the arc
module ring(outer, inner) {
	rotate_extrude($fn=64) translate([inner, 0, 0]) // rotate([0, 0, -22.5]) 
        printable_circle(r = outer, overhang = overhang_angle);
}

module bar(h, flip=false) {
//	angle = 180/bar_facets;
//	rotation = flip ? angle : -angle;
	//rotate([0, 0, -rotation]) 
        linear_extrude(h) rotate(180) printable_circle(r = bar_r, overhang = overhang_angle);
}

module printable_circle(r, overhang=45) {
    n = [r * cos(overhang), r * (1 - sin(overhang))];
    m = [n.x - n.y * tan(overhang), 0];
    circle(r);
    translate([0, -r]) polygon([-m, m, n, [-n.x, n.y]]);
}

module printable_sphere(r, overhang) {
    rotate_extrude() intersection() {
        printable_circle(r, overhang);
        translate([r, 0]) square(r*2, center=true);
    }
}


// trim the ring
module partial_ring(outer, inner, deg) {
    epsilon = 0.01;
    compliment = 360 - deg;
        
	difference() {
		ring(outer, inner);

		if (compliment < 90) {
			translate([0, 0, -outer * 2])
				linear_extrude(height = outer * 4)
					polygon(points = [[0,0],
														[-(inner + outer), 0], 
														[-(inner + outer),(inner + outer) * tan(compliment)]]);
		}
		else if (compliment < 180) {
			translate([-(inner + outer), 0, -outer * 2])
				cube(size = [inner + outer + epsilon, inner + outer + epsilon, outer*4], center = false);

			translate([0, 0, -outer * 2])
			linear_extrude(height = outer * 4)
				polygon(points = [[0,0],
													[0, (inner + outer)], 
													[(inner + outer) * tan(compliment - 90),
			(inner + outer),]]);
		}
		else if (compliment < 270) {
			translate([-(inner + outer), 0, -outer*2])
				cube(size = [(inner + outer)*2 + epsilon, inner + outer + epsilon, outer*4], center = false);

			translate([0, 0, -outer * 2])
			linear_extrude(height = outer * 4)
				polygon(points = [[0,0],
													[(inner + outer), 0], 
													[(inner + outer),
			-(inner + outer) * tan(compliment - 180)]]);

		}
		else {
			translate([-(inner + outer), 0, -outer * 2])
				cube(size = [(inner + outer)*2 + epsilon, inner + outer + epsilon, outer*4], center = false);

            translate([0, -(inner + outer), -outer*2])
                cube(size = [inner + outer + epsilon, inner + outer + epsilon, outer*4], center = false);
            
			translate([0, 0, -outer*2])
			linear_extrude(height = outer * 4)
				polygon(points = [[0,0],
													[0, -(inner + outer)], 
													[(inner + outer) * tan(90 - (compliment - 180)),
														-(inner + outer)]]);
		}
  }
}


//module hangerhook() {
//	straight_h = max(hook_h - bar_d - hook_d, 0);
//	top_offset = hook_h - hook_r - bar_d;
// 
//  rotate(a = [-90, 0, -90]) 
//	{
//		translate([0, 0, top_offset])
//            rotate(a = [90, 180, 0])
//            partial_ring(bar_r, hook_r, 180);
//		translate([-hook_r, 0, top_offset]) rotate([90, 0, 0])
//            printable_sphere(r = bar_d/2, overhang = overhang_angle);
//        bar(h=straight_h);
//        translate([0, 0, hook_r+straight_h]) rotate(a = [90, -90, 0])
//            partial_ring(bar_r, hook_r, 90);
//		translate([0, 0, straight_h]) 
//        rotate([90, 0, 0]) 
//            printable_sphere(r = bar_d/2, overhang = overhang_angle);
//	}
//}

module hangerhook(length, height, bar_d, bend_d, hook_d, hook_h) {
	straight_h = 45;// max(hook_h - bar_d - hook_d, 0);
	top_offset = hook_h - hook_r - 6.35;//bar_d;

 
  rotate(a = [-90, 0, -90]) 
	{
		// build the hook in the z direction
		// top half of hook
        translate([-hook_r-8, 0, hook_h-26]) rotate(a = [0, 45, 0]) bar(h=20);
		translate([0, 0, top_offset])
            rotate(a = [90, 220, 0])
            partial_ring(bar_r, hook_r, 175);
//		translate([-hook_r * sqrt(2)/2, 0, top_offset+hook_r* sqrt(2)/2]) rotate([0, 220, 0])
//            printable_sphere(r = bar_d/2, overhang = overhang_angle);
		// rounded end of top half

//		translate([-hook_r * sqrt(2)/2-8*cos(45), 0, top_offset+hook_r* sqrt(2)/2-8*cos(45)]) rotate([0, 220, 0])
//            printable_sphere(r = bar_d/2, overhang = overhang_angle);
        //translate([hook_r, 0, hook_r]) 
        bar(h=straight_h);
        translate([0, 0, straight_h]) rotate(a = [0, 36.5, 0]) bar(h=26);
		translate([26 * sin(36.5), 0, straight_h + 26 *cos(36.5)]) 
        rotate([0, 36, 0]) 
            printable_sphere(r = bar_d/2, overhang = overhang_angle);
            //partial_ring(bar_r, hook_r, 90);
    // smooth entry into the bracket
		translate([0, 0, straight_h]) 
        //scale([3, 1, 1]) 
        rotate([90, 0, 0]) 
            printable_sphere(r = bar_d/2, overhang = overhang_angle);
//		translate([0, 0, -bar_r / 2])
//			cube(size = [plate_h, thickness, bar_r], center = true);
//		// middle of "I"
//		translate([0, 0, -plate_w / 2 - tolerance - bar_r])
//			cube(size = [crossbeam_h, bar_r, plate_w + tolerance * 2], center = true);
//		// bottom of "I"
//		translate([0, 0, -(plate_w + bar_r * 3 / 2 + tolerance * 2)])
//			cube(size = [plate_h, thickness, bar_r], center = true);
	}
}

translate([0,0,-110]) rotate([0,-90,0]) {
translate([height,0,0])hangerhook(length, height, bar_d, bend_d, hook_d, hook_h);

translate([0,-bottom_l,0])
//translate([-sin(corner_a)*top_l - bend_d - bar_r, - bottom_l - (plate_h)/2 + 1, 0])
rotate(a = [270, 0, 0]) {

    // horizontal bar
    bar(h=bottom_l);
    //bar(h=bottom_l,flip=true);
    // diagonal bar
    translate([bend_r, 0, 0])
        rotate(a = [0, corner_a, 0])
        translate([bend_r, 0, 0])  
            bar(h=top_l + bar_r/2, flip=true);

    // outside bend
    translate([bend_r, 0, 0])
        rotate([90, 0, 0]) partial_ring(bar_r, bend_r, bend_a);
};

mirror([0,1,0]) {translate([0,-bottom_l,0])
//translate([-sin(corner_a)*top_l - bend_d - bar_r, - bottom_l - (plate_h)/2 + 1, 0])
rotate(a = [270, 0, 0]) {

    // horizontal bar
    bar(h=bottom_l);
    //bar(h=bottom_l,flip=true);
    // diagonal bar
    translate([bend_r, 0, 0])
        rotate(a = [0, corner_a, 0])
        translate([bend_r, 0, 0])  
            bar(h=top_l + bar_r/2, flip=true);

    // outside bend
    translate([bend_r, 0, 0])
        rotate([90, 0, 0]) partial_ring(bar_r, bend_r, bend_a);
}}
}
}

hanger(length, height, bar_d, bend_d, hook_d, hook_h);