include <loft.scad>;
use <dotSCAD/src/path_extrude.scad>;
use <dotSCAD/src/helix_extrude.scad>;

$fn=20;

// M4 turnbuckle size
center_length = 55;
center_width = 11;
center_height = 3.5;
cutout_length = 48;
cutout_width = 6.5;
bolt_length = 6;
bolt_width = 4;
bolt_inc_perc = 0.1;

center_n=5;
center_outr=5.5;
center_inr=3.5;

hook_thickness = 1.5;
hook_d = 10;
hook_width = 6;
hook_length = 7.5;

max_screw_length = 27 + bolt_length/2;

// module for the center of the turnbuckle
module center_shape_polygon(n, r, center_width) {
    step = 90/n;
    // top line
    p1 = [[0, center_length / 2]];
    // top right corner
    p2 = [for (t=[90:-step:0]) [r*cos(t), cutout_length/2 + r*sin(t)]];
    // right line
    p3 = [[center_width / 2, 0]];
    // bottom right corner
    p4 = [for (t=[0:-step:-90]) [r*cos(t), -cutout_length/2 + r*sin(t)]];
    // bottom line
    p5 = [[0, -center_length / 2]];
    // bottom left corner
    p6 = [for (t=[-90:-step:-180]) [r*cos(t), -cutout_length/2 + r*sin(t)]];
    // left line
    p7 = [[-center_width / 2, 0]];
    // top left corner
    p8 = [for (t=[-180:-step:-270]) [r*cos(t), cutout_length/2 + r*sin(t)]];
    center_points = concat(p1, p2, p3, p4, p5, p6, p7, p8);
    polygon(center_points);
}

// module for the "bolts" on the two sides
module bolt() {
    upper_points = [
        [bolt_length / 2, bolt_width / 2, bolt_length / 2],
        [bolt_length / 2, -bolt_width / 2, bolt_length / 2],
        [-bolt_length / 2, -bolt_width / 2, bolt_length / 2],
        [-bolt_length / 2, bolt_width / 2, bolt_length / 2],
    ];
    ratio = 1+bolt_inc_perc;
    lower_points = [
        [bolt_length*ratio / 2, bolt_width*ratio / 2, -bolt_length / 2],
        [bolt_length*ratio / 2, -bolt_width*ratio / 2, -bolt_length / 2],
        [-bolt_length*ratio / 2, -bolt_width*ratio / 2, -bolt_length / 2],
        [-bolt_length*ratio / 2, bolt_width*ratio / 2, -bolt_length / 2],
    ];
    loft(lower_points, upper_points, 1);
}

module hook() {
    n=20;
    step=360/n;
    shape_pts = [for (t=[0:step:360]) [hook_thickness*cos(t),hook_thickness*sin(t)]];

    // the base of the hook
    rotate([0,90,25]) translate([0,0,-hook_length/4-hook_width/20]) cylinder(hook_length, hook_thickness, hook_thickness, center=true);
    rotate([0,90,57]) translate([0,-hook_width/8,hook_length/2]) cylinder(hook_length/2, hook_thickness, hook_thickness, center=true);
    translate([hook_length/5,hook_thickness/2,0]) sphere(hook_thickness*1.05);
    
    // the circle portion of the hook
    cangle = 210;
    rangle = -50;
    hook_newr = hook_d/2 + hook_thickness/2;
    circle_base = [hook_d-hook_thickness+0.35, hook_thickness+0.45];
    translate(circle_base) rotate([0,0,rangle]) rotate_extrude(angle=cangle) translate([hook_newr,0]) polygon(shape_pts);
    
    // the tip of the hook
    tip_base = [circle_base[0] + hook_newr*cos(rangle), circle_base[1] + hook_newr*sin(rangle)];
    lower_center = [0,0,0];
    tip_thickness = 0.6;
    tip_pts = [for (t=[0:step:360]) [tip_thickness*cos(t),tip_thickness*sin(t)]];
    tip_length = 6;
    upper_center = [0,0,tip_length];
    lower_points = [for (i=[0:1:20]) [shape_pts[i][0], shape_pts[i][1], 0]];
    upper_points = [for (i=[0:1:20]) [tip_pts[i][0], tip_pts[i][1], tip_length]];
    translate(tip_base) rotate([90, 0, rangle]) loft(lower_points, upper_points, 1);
}

module screw(dir) {
    screw_length = max_screw_length;
    rotate([90,0,0]) union() {
        cylinder(h=screw_length,r=hook_thickness,center=true);
        ratio = 0.4;
        shape_pts = [
            [0,0],
            [hook_thickness*ratio,hook_thickness*ratio/3],
            [0,hook_thickness*ratio/3*2]
        ];        
        nl = 50;
        translate([0,0,-max_screw_length/2])
        helix_extrude(shape_pts, 
            radius = hook_thickness, 
            levels = nl,
            level_dist = screw_length/nl,
            vt_dir = dir
        );
    }
};

module M4turnbuckle(length_extended) {
    linear_extrude(height=center_height, center=true)
    difference() {
        center_shape_polygon(center_n, center_outr, center_width);
        center_shape_polygon(center_n, center_inr, cutout_width);
    };
    
    translate([0, (center_length + cutout_length) / 4 + bolt_length/2, 0]) rotate([-90, 0, 0]) bolt();
    translate([0, -(center_length + cutout_length) / 4 - bolt_length/2, 0]) rotate([90, 0, 0]) bolt();
    
    translate([0, length_extended / 2, 0]) union() {
        translate([0, (center_length + cutout_length) / 4 + bolt_length*1.5, 0]) rotate([0,0,65]) hook();
        translate([0, (center_length + bolt_length*1.5 - max_screw_length) / 2, 0]) screw("SPI_DOWN");
    };

    translate([0, -length_extended / 2, 0]) union() {
        translate([0, -((center_length + cutout_length) / 4 + bolt_length*1.5), 0]) rotate([0,0,245]) hook();
        translate([0, -(center_length + bolt_length*1.5 - max_screw_length) / 2, 0]) screw("SPI_UP");
    };
    echo(-(center_length + bolt_length*1.5 - max_screw_length) / 2);
};

length_extended = 0; // no extension
//length_extended = 45.7; // full extension
//length_extended = 20;
M4turnbuckle(length_extended);